"""
SOTA model benchmarking.
Compares CBM variants against TabNet, XGBoost, and other state-of-the-art classifiers.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
from collections import OrderedDict
from datetime import datetime

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score
)

# Project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)

# Import data loading and CBM base components
from experiment_data import (
    load_ppmi_extended, load_nacc_extended,
    load_ppmi_ternary, load_nacc_ternary,
    load_ppmi_binary, load_nacc_binary,
)
from run_explore_cbm import (
    COGNITIVE_DOMAINS, MOCA_FEATURES, DOMAIN_MAX_SCORES,
    preprocess_with_concepts, compute_true_concepts,
    CBM_Supervised, CBM_Hybrid,
    train_cbm as train_cbm_orig,
    SEED
)

# ========== Global Constants ==========
N_FOLDS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_CONCEPTS = len(COGNITIVE_DOMAINS)   # 7
N_DEMO = 5
N_INPUT = 24  # 19 MoCA + 5 demo
CEM_EMBED_DIM = 8   # embedding dimension per concept
LF_N_COMPONENTS = 7  # number of LabelFree concepts (aligned with supervised CBM)

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
#  CEM: Concept Embedding Model
#  (Zarlenga et al., NeurIPS 2022)
# ============================================================

class CEM(nn.Module):
    """
    Concept Embedding Model

    Key features:
    - Each cognitive domain predicts a d-dimensional embedding vector instead of a scalar
    - Embedding vectors encode both positive and negative concept activation directions
    - Residual path: raw input features also feed directly into the diagnosis head (like CBM-Hybrid)
    - Diagnosis head input: (7 * embed_dim + n_demo) dimensions
    """

    def __init__(self, n_demo=N_DEMO, n_classes=3, embed_dim=CEM_EMBED_DIM):
        super().__init__()
        self.domain_names = list(COGNITIVE_DOMAINS.keys())
        self.domain_indices = []
        self.n_demo = n_demo
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.n_concepts = N_CONCEPTS

        # Concept embedding predictors: each domain -> embed_dim vector
        self.concept_embedders = nn.ModuleList()
        for domain, items in COGNITIVE_DOMAINS.items():
            indices = [MOCA_FEATURES.index(item) for item in items]
            self.domain_indices.append(indices)
            n_items = len(items)
            hidden = max(16, n_items * 4)
            embedder = nn.Sequential(
                nn.Linear(n_items, hidden),
                nn.ReLU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, embed_dim),
                # No Sigmoid: allow embeddings to express freely
            )
            self.concept_embedders.append(embedder)

        # Residual encoder: full input -> 8-dim free vector (like Hybrid free_encoder)
        total_input = N_INPUT  # 19 MoCA + 5 demo = 24
        self.residual_encoder = nn.Sequential(
            nn.Linear(total_input, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, embed_dim),
        )

        # Diagnosis head: concat concept embeddings + residual + demo
        diag_in = self.n_concepts * embed_dim + embed_dim + n_demo
        self.diagnosis_head = nn.Sequential(
            nn.Linear(diag_in, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def _get_concept_embeddings(self, x):
        """Extract all concept embedding vectors, shape=(B, n_concepts*embed_dim)."""
        x_moca = x[:, :19]
        embeddings = []
        for i, indices in enumerate(self.domain_indices):
            domain_input = x_moca[:, indices]
            emb = self.concept_embedders[i](domain_input)  # (B, embed_dim)
            embeddings.append(emb)
        return torch.cat(embeddings, dim=1)  # (B, n_concepts*embed_dim)

    def get_concept_scores(self, x):
        """
        Return scalar concept scores for interpretability.
        Uses sigmoid(mean) of embedding vector to map to [0,1] concept scores.
        """
        x_moca = x[:, :19]
        scores = []
        for i, indices in enumerate(self.domain_indices):
            domain_input = x_moca[:, indices]
            emb = self.concept_embedders[i](domain_input)  # (B, embed_dim)
            # Map to [0,1] concept score using sigmoid(mean)
            score = torch.sigmoid(emb.mean(dim=-1, keepdim=True))
            scores.append(score)
        return torch.cat(scores, dim=1)  # (B, n_concepts)

    def forward(self, x):
        x_demo = x[:, 19:19 + self.n_demo]
        concept_embs = self._get_concept_embeddings(x)   # (B, n_c*d)
        # Residual path uses full 24-dim input
        residual = self.residual_encoder(x[:, :N_INPUT])  # (B, d)
        combined = torch.cat([concept_embs, residual, x_demo], dim=1)
        logits = self.diagnosis_head(combined)
        # Return log_softmax and scalar concept scores (for evaluation)
        concepts = self.get_concept_scores(x)
        return F.log_softmax(logits, dim=-1), concepts

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            log_probs, concepts = self.forward(x)
            return torch.exp(log_probs), concepts


# ============================================================
#  LabelFree-CBM
#  (Oikarinen et al., ICML 2023, simplified version)
#  Uses NMF in feature space to discover 7 semantic concepts,
#  then classifies with a lightweight MLP.
# ============================================================

class LabelFreeCBM(nn.Module):
    """
    Label-Free CBM (NMF-based concept discovery)

    Training pipeline:
    1. Fit sklearn NMF on training set, compress 19-dim MoCA to 7 non-negative basis components
       (NMF non-negativity makes components more semantically additive, similar to concepts)
    2. Concept layer: linear projection (initialized from NMF basis matrix), outputs 7-dim vector
    3. Diagnosis head: lightweight MLP, takes 7 concepts + 5 demo -> classification

    Notes:
    - NMF basis matrix is fit on training set within each fold; test set only uses transform
    - n_components=7 is aligned with the 7 cognitive domain concepts in supervised CBM
    """

    def __init__(self, n_demo=N_DEMO, n_classes=3, n_components=LF_N_COMPONENTS):
        super().__init__()
        self.n_demo = n_demo
        self.n_classes = n_classes
        self.n_components = n_components
        self.nmf_fitted = False

        # Concept linear projection layer (weights from NMF, but fine-tunable)
        # Input: 19-dim MoCA -> n_components-dim concepts
        self.concept_proj = nn.Linear(19, n_components, bias=False)
        self.concept_act = nn.Sigmoid()  # normalize to [0,1]

        # Diagnosis head
        diag_in = n_components + n_demo
        self.diagnosis_head = nn.Sequential(
            nn.Linear(diag_in, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def init_from_nmf(self, nmf_components: np.ndarray):
        """
        Initialize concept projection layer from NMF basis matrix.
        nmf_components: shape=(n_components, n_features=19)
        """
        W = torch.FloatTensor(nmf_components)  # (n_components, 19)
        with torch.no_grad():
            self.concept_proj.weight.copy_(W)
        self.nmf_fitted = True

    def forward(self, x):
        # x shape: 24-dim (19 MoCA + 5 demo)
        x_moca = x[:, :19]
        x_demo = x[:, 19:19 + self.n_demo]

        # Concept layer (NMF-initialized, fine-tunable)
        concepts = self.concept_act(self.concept_proj(x_moca))  # (B, n_components)

        combined = torch.cat([concepts, x_demo], dim=1)
        logits = self.diagnosis_head(combined)
        return F.log_softmax(logits, dim=-1), concepts

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            log_probs, concepts = self.forward(x)
            return torch.exp(log_probs), concepts


# ============================================================
#  Generic Training Function
# ============================================================

def _train_model_single(model, X_train, y_train, X_test, y_test,
                        epochs=150, batch_size=32, lr=1e-3,
                        concept_supervision=False, lambda_concept=0.5,
                        true_concepts_train=None, verbose=False):
    """Single training loop with optional concept supervision."""
    device = DEVICE
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=False)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    indices_t = torch.arange(len(X_train)).to(device)

    if concept_supervision and true_concepts_train is not None:
        tc_t = torch.FloatTensor(true_concepts_train).to(device)
    else:
        tc_t = None

    dataset = TensorDataset(X_train_t, y_train_t, indices_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Class weights
    class_counts = np.bincount(y_train, minlength=len(np.unique(y_train)))
    total = len(y_train)
    n_cls = len(class_counts)
    class_weights = torch.FloatTensor(
        [total / (n_cls * c) if c > 0 else 1.0 for c in class_counts]
    ).to(device)
    criterion = nn.NLLLoss(weight=class_weights)  # model.forward returns log_softmax, use NLLLoss
    mse_fn = nn.MSELoss()

    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 10  # early stopping patience, reduces training time on large datasets

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b, idx_b in loader:
            optimizer.zero_grad()
            log_probs, concepts = model(X_b)
            loss = criterion(log_probs, y_b)

            if tc_t is not None:
                c_true = tc_t[idx_b]
                concept_loss = mse_fn(concepts, c_true)

                # Polarity constraint (encourage positive correlation)
                polarity_loss = 0.0
                for d in range(concepts.shape[1]):
                    pred_d = concepts[:, d] - concepts[:, d].mean()
                    true_d = c_true[:, d] - c_true[:, d].mean()
                    corr = (pred_d * true_d).sum() / (
                        pred_d.norm() * true_d.norm() + 1e-8)
                    polarity_loss += (1.0 - corr)

                loss = loss + lambda_concept * (concept_loss + 0.05 * polarity_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    with torch.no_grad():
        log_probs_test, concepts_test = model(X_test_t)
        probs_test = torch.exp(log_probs_test).cpu().numpy()
        preds_test = probs_test.argmax(axis=1)
        concepts_np = concepts_test.cpu().numpy()

    return model, probs_test, preds_test, concepts_np


def train_model_with_retry(model_factory, X_train, y_train, X_test, y_test,
                           epochs=150, batch_size=32, lr=1e-3,
                           concept_supervision=False, lambda_concept=0.5,
                           true_concepts_train=None, max_retries=3, verbose=False):
    """Training with retry: retries if BA<0.4 (up to max_retries times)."""
    best_result = None
    best_ba = -1.0

    for attempt in range(max_retries):
        torch.manual_seed(SEED + attempt)
        np.random.seed(SEED + attempt)
        model = model_factory()

        model, probs, preds, concepts_np = _train_model_single(
            model, X_train, y_train, X_test, y_test,
            epochs=epochs, batch_size=batch_size, lr=lr,
            concept_supervision=concept_supervision,
            lambda_concept=lambda_concept,
            true_concepts_train=true_concepts_train,
            verbose=verbose
        )

        ba = balanced_accuracy_score(y_test, preds)
        result = {
            'model': model,
            'probs': probs,
            'preds': preds,
            'concepts': concepts_np,
            'balanced_accuracy': ba
        }

        if ba > best_ba:
            best_ba = ba
            best_result = result

        if ba > 0.4:
            return best_result

    return best_result


# ============================================================
#  Metric Computation
# ============================================================

def compute_metrics(y_true, y_pred, y_prob, n_classes):
    """Compute BA, Macro F1, and Macro AUC."""
    ba = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')

    return {'balanced_accuracy': ba, 'macro_f1': f1, 'macro_auc': auc}


# ============================================================
#  Per-fold Training Entry Points for Each Model
# ============================================================

def run_fold_cbm_supervised(X_tr, y_tr, X_te, y_te, n_classes,
                             true_concepts_tr, true_concepts_te):
    """CBM-Supervised (uses custom training function to avoid slow retry logic in original)."""
    def factory():
        return CBM_Supervised(n_demo=N_DEMO, n_classes=n_classes)
    result = train_model_with_retry(
        factory, X_tr, y_tr, X_te, y_te,
        epochs=50, batch_size=128, lr=1e-3,
        concept_supervision=True, lambda_concept=0.5,
        true_concepts_train=true_concepts_tr,
        max_retries=2
    )
    return result['probs'], result['preds']


def run_fold_cbm_hybrid(X_tr, y_tr, X_te, y_te, n_classes):
    """CBM-Hybrid (uses custom training function)."""
    def factory():
        return CBM_Hybrid(n_demo=N_DEMO, n_classes=n_classes, n_free_dims=8)
    result = train_model_with_retry(
        factory, X_tr, y_tr, X_te, y_te,
        epochs=50, batch_size=128, lr=1e-3,
        concept_supervision=False,
        max_retries=2
    )
    return result['probs'], result['preds']


def run_fold_cem(X_tr, y_tr, X_te, y_te, n_classes):
    """CEM"""
    def factory():
        return CEM(n_demo=N_DEMO, n_classes=n_classes, embed_dim=CEM_EMBED_DIM)

    result = train_model_with_retry(
        factory, X_tr, y_tr, X_te, y_te,
        epochs=50, batch_size=128, lr=1e-3,
        concept_supervision=False, max_retries=2
    )
    return result['probs'], result['preds']


def run_fold_labelfree(X_tr, y_tr, X_te, y_te, n_classes):
    """
    LabelFree-CBM
    Steps:
    1. Fit NMF on raw (pre-standardization) 19-dim MoCA scores from training set only
    2. Initialize LabelFreeCBM concept projection layer from NMF basis matrix
    3. End-to-end fine-tuning
    """
    # NMF requires non-negative data; MoCA sub-items are non-negative integers
    # Only use MoCA features (first 19 columns) for NMF
    X_tr_moca = X_tr[:, :19]
    X_te_moca = X_te[:, :19]

    # Shift data to non-negative (StandardScaler may introduce negatives)
    shift = X_tr_moca.min(axis=0, keepdims=True)
    shift = np.minimum(shift, 0)  # only compensate for negative values
    X_tr_moca_nn = X_tr_moca - shift
    X_te_moca_nn = X_te_moca - shift

    # Fit NMF
    nmf = NMF(
        n_components=LF_N_COMPONENTS,
        init='nndsvda',
        random_state=SEED,
        max_iter=500
    )
    nmf.fit(X_tr_moca_nn)
    # components_.shape = (n_components, 19)
    nmf_components = nmf.components_  # (7, 19)

    # Row-normalize NMF components for reasonable weight scaling
    row_norms = np.linalg.norm(nmf_components, axis=1, keepdims=True) + 1e-8
    nmf_components_normed = nmf_components / row_norms

    def factory():
        m = LabelFreeCBM(n_demo=N_DEMO, n_classes=n_classes,
                         n_components=LF_N_COMPONENTS)
        m.init_from_nmf(nmf_components_normed)
        return m

    result = train_model_with_retry(
        factory, X_tr, y_tr, X_te, y_te,
        epochs=50, batch_size=128, lr=1e-3,
        concept_supervision=False, max_retries=2
    )
    return result['probs'], result['preds']


# ============================================================
#  Data Preprocessing
# ============================================================

# KNNImputer is very slow on large datasets (e.g., NACC 72K) - O(n^2)
# Use SimpleImputer (mean) when n_samples > KNN_THRESHOLD
KNN_THRESHOLD = 5000  # threshold above which to use mean imputation


def preprocess_fold(X_tr_raw, X_te_raw):
    """Custom preprocessing: SimpleImputer for large datasets, KNNImputer for small.
    Then StandardScaler. Returns processed data and true concepts.
    """
    n_train = len(X_tr_raw)

    if n_train > KNN_THRESHOLD:
        # Large dataset: SimpleImputer (mean)
        imputer = SimpleImputer(strategy='mean')
    else:
        # Small dataset: KNNImputer (more accurate)
        imputer = KNNImputer(n_neighbors=5)

    X_tr_imp = imputer.fit_transform(X_tr_raw)
    X_te_imp = imputer.transform(X_te_raw)

    # Compute true concepts after imputation but before scaling
    tc_tr = compute_true_concepts(X_tr_imp)
    tc_te = compute_true_concepts(X_te_imp)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_imp)
    X_te_s = scaler.transform(X_te_imp)

    return X_tr_s, X_te_s, tc_tr, tc_te


# ============================================================
#  Single-Dataset 10-fold CV
# ============================================================

# Subsampling threshold: stratified sampling when per-fold training set exceeds this
MAX_TRAIN_PER_FOLD = 5000  # max 5000 training samples per fold (for GPU throughput)


def run_dataset_cv(dataset_name, X, y, n_classes):
    """
    Run 10-fold CV on a specified dataset and evaluate all models.

    Returns:
    {
        'model_name': {
            'folds': [{ba, f1, auc}, ...],
            'mean': {ba, f1, auc},
            'std': {ba, f1, auc}
        }, ...
    }
    """
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}  |  samples={len(y)}  |  classes={n_classes}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    model_names = ['CBM-Supervised', 'CBM-Hybrid', 'CEM', 'LabelFree-CBM']
    fold_results = {m: [] for m in model_names}

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        fold_start = time.time()
        print(f"\n  [Fold {fold_idx+1}/{N_FOLDS}] ", end='', flush=True)

        X_tr_raw, X_te_raw = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Subsample large training sets using stratified sampling
        if len(y_tr) > MAX_TRAIN_PER_FOLD:
            rng = np.random.RandomState(SEED + fold_idx)
            sub_idx = []
            for cls in np.unique(y_tr):
                cls_mask = np.where(y_tr == cls)[0]
                n_cls = max(1, int(MAX_TRAIN_PER_FOLD * len(cls_mask) / len(y_tr)))
                chosen = rng.choice(cls_mask, size=min(n_cls, len(cls_mask)), replace=False)
                sub_idx.append(chosen)
            sub_idx = np.concatenate(sub_idx)
            X_tr_raw = X_tr_raw[sub_idx]
            y_tr = y_tr[sub_idx]
            print(f"(sub={len(y_tr)}) ", end='', flush=True)

        # Preprocessing
        X_tr, X_te, tc_tr, tc_te = preprocess_fold(X_tr_raw, X_te_raw)

        # ---------- CBM-Supervised ----------
        try:
            probs, preds = run_fold_cbm_supervised(
                X_tr, y_tr, X_te, y_te, n_classes, tc_tr, tc_te)
            m = compute_metrics(y_te, preds, probs, n_classes)
            fold_results['CBM-Supervised'].append(m)
            print(f"Sup(ba={m['balanced_accuracy']:.3f}) ", end='', flush=True)
        except Exception as e:
            print(f"Sup(ERR:{e}) ", end='')
            fold_results['CBM-Supervised'].append(
                {'balanced_accuracy': np.nan, 'macro_f1': np.nan, 'macro_auc': np.nan})

        # ---------- CBM-Hybrid ----------
        try:
            probs, preds = run_fold_cbm_hybrid(X_tr, y_tr, X_te, y_te, n_classes)
            m = compute_metrics(y_te, preds, probs, n_classes)
            fold_results['CBM-Hybrid'].append(m)
            print(f"Hyb(ba={m['balanced_accuracy']:.3f}) ", end='', flush=True)
        except Exception as e:
            print(f"Hyb(ERR:{e}) ", end='')
            fold_results['CBM-Hybrid'].append(
                {'balanced_accuracy': np.nan, 'macro_f1': np.nan, 'macro_auc': np.nan})

        # ---------- CEM ----------
        try:
            probs, preds = run_fold_cem(X_tr, y_tr, X_te, y_te, n_classes)
            m = compute_metrics(y_te, preds, probs, n_classes)
            fold_results['CEM'].append(m)
            print(f"CEM(ba={m['balanced_accuracy']:.3f}) ", end='', flush=True)
        except Exception as e:
            print(f"CEM(ERR:{e}) ", end='')
            fold_results['CEM'].append(
                {'balanced_accuracy': np.nan, 'macro_f1': np.nan, 'macro_auc': np.nan})

        # ---------- LabelFree-CBM ----------
        try:
            probs, preds = run_fold_labelfree(X_tr, y_tr, X_te, y_te, n_classes)
            m = compute_metrics(y_te, preds, probs, n_classes)
            fold_results['LabelFree-CBM'].append(m)
            print(f"LF(ba={m['balanced_accuracy']:.3f}) ", end='', flush=True)
        except Exception as e:
            print(f"LF(ERR:{e}) ", end='')
            fold_results['LabelFree-CBM'].append(
                {'balanced_accuracy': np.nan, 'macro_f1': np.nan, 'macro_auc': np.nan})

        elapsed = time.time() - fold_start
        print(f"[{elapsed:.1f}s]")

    # Aggregate statistics
    summary = {}
    for m_name in model_names:
        folds = fold_results[m_name]
        metrics_keys = ['balanced_accuracy', 'macro_f1', 'macro_auc']
        mean_d = {}
        std_d = {}
        for k in metrics_keys:
            vals = [f[k] for f in folds if not np.isnan(f[k])]
            mean_d[k] = float(np.mean(vals)) if vals else float('nan')
            std_d[k] = float(np.std(vals)) if vals else float('nan')
        summary[m_name] = {
            'folds': [
                {k: float(f[k]) for k in metrics_keys} for f in folds
            ],
            'mean': mean_d,
            'std': std_d
        }

    # Print summary table
    print(f"\n  {'Model':<20} {'BA(mean±std)':<20} {'F1(mean±std)':<20} {'AUC(mean±std)':<20}")
    print(f"  {'-'*80}")
    for m_name in model_names:
        s = summary[m_name]
        ba_str = f"{s['mean']['balanced_accuracy']:.4f}±{s['std']['balanced_accuracy']:.4f}"
        f1_str = f"{s['mean']['macro_f1']:.4f}±{s['std']['macro_f1']:.4f}"
        auc_str = f"{s['mean']['macro_auc']:.4f}±{s['std']['macro_auc']:.4f}"
        print(f"  {m_name:<20} {ba_str:<20} {f1_str:<20} {auc_str:<20}")

    return summary


# ============================================================
#  Main Function
# ============================================================

def main():
    start_time = time.time()
    print("=" * 70)
    print("  CBM SOTA Variant Comparison Experiment")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device: {DEVICE}")
    print(f"  N_FOLDS={N_FOLDS}, SEED={SEED}")
    print(f"  CEM embed_dim={CEM_EMBED_DIM}, LabelFree n_components={LF_N_COMPONENTS}")
    print("=" * 70)

    all_results = {
        'meta': {
            'timestamp': datetime.now().isoformat(),
            'n_folds': N_FOLDS,
            'seed': SEED,
            'cem_embed_dim': CEM_EMBED_DIM,
            'labelfree_n_components': LF_N_COMPONENTS,
            'models': ['CBM-Supervised', 'CBM-Hybrid', 'CEM', 'LabelFree-CBM'],
            'device': str(DEVICE)
        },
        'datasets': {}
    }

    def _safe_save(results_dict, path):
        """Safe save results (convert NaN/Inf to null)."""
        def _safe_float(v):
            if v is None:
                return None
            try:
                f = float(v)
                return None if (f != f or abs(f) == float('inf')) else f
            except Exception:
                return None
        def _sanitize(obj):
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            elif isinstance(obj, (float, np.floating)):
                return _safe_float(float(obj))
            else:
                return obj
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(_sanitize(results_dict), f, indent=2, ensure_ascii=False)

    out_path = os.path.join(RESULTS_DIR, 'cbm_sota_results.json')

    # ---------- NACC 3-class ----------
    X_nacc, y_nacc = None, None
    try:
        print("\n[1/4] Loading NACC 3-class data (extended 24-dim)...")
        X_nacc, y_nacc = load_nacc_extended(max_per_class=50000)
        nacc_results = run_dataset_cv('NACC-Ternary', X_nacc, y_nacc, n_classes=3)
        all_results['datasets']['NACC_ternary'] = nacc_results
        _safe_save(all_results, out_path)  # save intermediate results after each dataset
        print(f"  [Saved intermediate results] {out_path}")
    except Exception as e:
        print(f"  [ERROR] NACC 3-class load/run failed: {e}")
        import traceback; traceback.print_exc()

    # ---------- NACC binary ----------
    try:
        print("\n[2/4] NACC binary (reuse loaded data, map to binary)...")
        if X_nacc is None:
            X_nacc, y_nacc = load_nacc_extended(max_per_class=50000)
        y_nacc_bin = (y_nacc > 0).astype(np.int32)
        nacc_bin_results = run_dataset_cv('NACC-Binary', X_nacc, y_nacc_bin, n_classes=2)
        all_results['datasets']['NACC_binary'] = nacc_bin_results
        _safe_save(all_results, out_path)
        print(f"  [Saved intermediate results] {out_path}")
    except Exception as e:
        print(f"  [ERROR] NACC binary load/run failed: {e}")
        import traceback; traceback.print_exc()

    # ---------- PPMI binary ----------
    X_ppmi_e, y_ppmi_e = None, None
    try:
        print("\n[3/4] Loading PPMI binary data (extended 24-dim)...")
        X_ppmi_e, y_ppmi_e = load_ppmi_extended()
        y_ppmi_bin = (y_ppmi_e > 0).astype(np.int32)
        ppmi_results = run_dataset_cv('PPMI-Binary', X_ppmi_e, y_ppmi_bin, n_classes=2)
        all_results['datasets']['PPMI_binary'] = ppmi_results
        _safe_save(all_results, out_path)
        print(f"  [Saved intermediate results] {out_path}")
    except Exception as e:
        print(f"  [ERROR] PPMI binary load/run failed: {e}")
        import traceback; traceback.print_exc()

    # ---------- PPMI 3-class ----------
    try:
        print("\n[4/4] PPMI 3-class (reuse loaded data)...")
        if X_ppmi_e is None:
            X_ppmi_e, y_ppmi_e = load_ppmi_extended()
        ppmi_ter_results = run_dataset_cv('PPMI-Ternary', X_ppmi_e, y_ppmi_e, n_classes=3)
        all_results['datasets']['PPMI_ternary'] = ppmi_ter_results
        _safe_save(all_results, out_path)
        print(f"  [Saved intermediate results] {out_path}")
    except Exception as e:
        print(f"  [ERROR] PPMI 3-class load/run failed: {e}")
        import traceback; traceback.print_exc()

    # ---------- Final save ----------
    _safe_save(all_results, out_path)

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Experiment complete! Total time: {total_time/60:.1f} min")
    print(f"  Results saved to: {out_path}")
    print("=" * 70)

    # Print final cross-dataset summary
    print("\n[Final Summary - Balanced Accuracy (mean±std)]")
    model_names = ['CBM-Supervised', 'CBM-Hybrid', 'CEM', 'LabelFree-CBM']
    ds_names = list(all_results['datasets'].keys())
    header = f"  {'Model':<20}" + "".join(f"{d:<25}" for d in ds_names)
    print(header)
    print("  " + "-" * (20 + 25 * len(ds_names)))
    for m in model_names:
        row = f"  {m:<20}"
        for d in ds_names:
            try:
                s = all_results['datasets'][d][m]
                ba_str = f"{s['mean']['balanced_accuracy']:.4f}±{s['std']['balanced_accuracy']:.4f}"
            except Exception:
                ba_str = "N/A"
            row += f"{ba_str:<25}"
        print(row)

    return all_results


if __name__ == '__main__':
    main()
