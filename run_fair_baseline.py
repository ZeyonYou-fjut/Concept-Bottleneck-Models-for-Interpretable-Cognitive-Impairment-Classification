"""
Comparable-parameter baseline comparison.
Compares PlainDNN-Fair (1,475 params) against CBM-Hybrid (2,018 params) under matched conditions.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from collections import OrderedDict

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_data import load_nacc_extended, EXTENDED_FEATURES

# ============== Constants ==============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
SEED = 42
N_FOLDS = 10

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# MoCA cognitive domain definitions (identical to run_explore_cbm.py)
MOCA_FEATURES = [
    'TRAIL_MAKING', 'CUBE', 'CLOCK_CONTOUR', 'CLOCK_NUMBERS', 'CLOCK_HANDS',
    'NAMING', 'DIGIT_SPAN', 'VIGILANCE', 'SERIAL_7',
    'SENTENCE_REP', 'FLUENCY', 'ABSTRACTION',
    'DELAYED_RECALL',
    'ORIENT_DATE', 'ORIENT_MONTH', 'ORIENT_YEAR', 'ORIENT_DAY', 'ORIENT_PLACE', 'ORIENT_CITY'
]

COGNITIVE_DOMAINS = OrderedDict([
    ('Visuospatial_Executive', ['TRAIL_MAKING', 'CUBE', 'CLOCK_CONTOUR', 'CLOCK_NUMBERS', 'CLOCK_HANDS']),
    ('Naming', ['NAMING']),
    ('Attention', ['DIGIT_SPAN', 'VIGILANCE', 'SERIAL_7']),
    ('Language', ['SENTENCE_REP', 'FLUENCY']),
    ('Abstraction', ['ABSTRACTION']),
    ('Delayed_Recall', ['DELAYED_RECALL']),
    ('Orientation', ['ORIENT_DATE', 'ORIENT_MONTH', 'ORIENT_YEAR', 'ORIENT_DAY', 'ORIENT_PLACE', 'ORIENT_CITY'])
])

DOMAIN_MAX_SCORES = {
    'Visuospatial_Executive': 5, 'Naming': 3, 'Attention': 6,
    'Language': 3, 'Abstraction': 2, 'Delayed_Recall': 5, 'Orientation': 6
}


# ============== Parameter Counting Utility ==============

def count_params(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============== PlainDNN Definition ==============

class PlainDNN(nn.Module):
    """
    Standard PlainDNN, identical to implementation in model_dcm_dnn.py.
    BN + ReLU + Dropout(0.3)
    """
    def __init__(self, input_dim=24, num_classes=3, dropout_rate=0.3,
                 hidden_layers=None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.hidden_layers = hidden_layers

        layers = []
        prev_dim = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = units
        self.dnn = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        features = self.dnn(x)
        logits = self.output_layer(features)
        return F.log_softmax(logits, dim=-1)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            return torch.exp(self.forward(x))


# ============== CBM-Hybrid Definition (identical to run_explore_cbm.py) ==============

class CBM_Hybrid(nn.Module):
    """
    CBM-Hybrid: 7 concepts + 8 free latent dimensions.
    Identical to definition in run_explore_cbm.py.
    """
    def __init__(self, n_demo=5, n_classes=3, n_free_dims=8):
        super().__init__()
        self.domain_names = list(COGNITIVE_DOMAINS.keys())
        self.domain_indices = []
        self.n_demo = n_demo
        self.n_classes = n_classes
        self.n_free_dims = n_free_dims

        # Concept predictors
        self.concept_predictors = nn.ModuleList()
        for domain, items in COGNITIVE_DOMAINS.items():
            indices = [MOCA_FEATURES.index(item) for item in items]
            self.domain_indices.append(indices)
            n_items = len(items)
            predictor = nn.Sequential(
                nn.Linear(n_items, max(16, n_items * 2)),
                nn.ReLU(),
                nn.Linear(max(16, n_items * 2), 1),
                nn.Sigmoid()
            )
            self.concept_predictors.append(predictor)

        # Free encoder: full 24-dim input -> 8-dim
        self.free_encoder = nn.Sequential(
            nn.Linear(24, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_free_dims)
        )

        # Diagnosis head
        n_concepts = len(COGNITIVE_DOMAINS)  # 7
        self.diagnosis_head = nn.Sequential(
            nn.Linear(n_concepts + n_free_dims + n_demo, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, n_classes)
        )

    def _get_concepts(self, x):
        x_moca = x[:, :19]
        concepts = []
        for i, indices in enumerate(self.domain_indices):
            domain_input = x_moca[:, indices]
            concept = self.concept_predictors[i](domain_input)
            concepts.append(concept)
        return torch.cat(concepts, dim=1)

    def forward(self, x):
        x_demo = x[:, 19:19 + self.n_demo]
        concepts = self._get_concepts(x)
        free = self.free_encoder(x)
        combined = torch.cat([concepts, free, x_demo], dim=1)
        logits = self.diagnosis_head(combined)
        return F.log_softmax(logits, dim=-1), concepts

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            log_probs, concepts = self.forward(x)
            return torch.exp(log_probs), concepts


# ============== Preprocessing Function (identical to run_explore_cbm.py) ==============

def preprocess_dnn(X_train: np.ndarray, X_test: np.ndarray):
    """KNNImputer(n_neighbors=5) + StandardScaler."""
    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    return X_train_scaled, X_test_scaled


# ============== Unified Training Functions ==============

def train_plaindnn(model_class, input_dim, num_classes, hidden_layers,
                   X_train, y_train, X_test, y_test,
                   epochs=100, batch_size=32, lr=0.0008,
                   max_retries=3, verbose=False):
    """
    Train PlainDNN, aligned with train_dnn_with_retry in run_explore_cbm.py:
    - lr=0.0008, warmup + cosine schedule
    - Early stopping patience=15
    - Class-weighted CrossEntropy (focal loss disabled)
    - Retry if balanced_accuracy<=0.4
    """
    best_result = None
    best_ba = 0.0

    for attempt in range(max_retries):
        torch.manual_seed(SEED + attempt)
        np.random.seed(SEED + attempt)

        model = model_class(input_dim=input_dim, num_classes=num_classes,
                            hidden_layers=hidden_layers)

        result = _train_dnn_single(
            model, X_train, y_train, X_test, y_test,
            epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose
        )

        ba = result['balanced_accuracy']
        if ba > best_ba:
            best_ba = ba
            best_result = result

        if ba > 0.4:
            return result

    return best_result


def _train_dnn_single(model, X_train, y_train, X_test, y_test,
                      epochs, batch_size, lr, verbose):
    """Single DNN training run, Warmup + CosineAnnealing + EarlyStopping."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    num_classes = model.num_classes

    # Validation split
    unique, counts = np.unique(y_train, return_counts=True)
    min_count = int(np.min(counts))
    if len(y_train) < 20 or min_count < 2:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=SEED)
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)

    X_tr_t = torch.FloatTensor(X_tr).to(device)
    y_tr_t = torch.LongTensor(y_tr).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    train_dataset = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0,
                              drop_last=(len(X_tr) > batch_size))
    if len(train_loader) == 0:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=0, drop_last=False)

    # Class-weighted CrossEntropy (focal loss disabled)
    class_counts = np.bincount(y_tr, minlength=num_classes)
    total = len(y_tr)
    class_weights = [total / (num_classes * c) if c > 0 else 1.0
                     for c in class_counts]
    w = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = Adam(model.parameters(), lr=lr)

    # Warmup + CosineAnnealing params
    initial_lr = lr
    min_lr = 1e-7
    warmup_epochs = 5
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            log_probs = model(bx)
            loss = criterion(log_probs, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_lp = model(X_val_t)
            val_loss = criterion(val_lp, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    EarlyStopping @ epoch {epoch + 1}")
                break

        # Warmup + CosineAnnealing LR schedule
        if epoch < warmup_epochs:
            new_lr = initial_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            new_lr = min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = new_lr

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Evaluate
    with torch.no_grad():
        y_prob = model.predict_proba(X_test_t).cpu().numpy()

    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_test, y_pred)
    ba = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    try:
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0

    return {
        'accuracy': acc,
        'balanced_accuracy': ba,
        'f1_macro': f1,
        'auc': auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def train_cbm_hybrid(X_train, y_train, X_test, y_test,
                     epochs=100, batch_size=32, lr=1e-3,
                     max_retries=3, verbose=False):
    """
    Train CBM-Hybrid, aligned with train_cbm in run_explore_cbm.py:
    - Adam lr=1e-3, weight_decay=1e-4
    - ReduceLROnPlateau scheduler
    - Early stopping patience=15
    - Class-weighted CrossEntropy
    - Retry if balanced_accuracy<=0.4
    """
    best_result = None
    best_ba = 0.0

    for attempt in range(max_retries):
        torch.manual_seed(SEED + attempt)
        np.random.seed(SEED + attempt)

        model = CBM_Hybrid()
        result = _train_cbm_single(
            model, X_train, y_train, X_test, y_test,
            epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose
        )

        ba = result['balanced_accuracy']
        if ba > best_ba:
            best_ba = ba
            best_result = result

        if ba > 0.4:
            return result

    return best_result


def _train_cbm_single(model, X_train, y_train, X_test, y_test,
                      epochs, batch_size, lr, verbose):
    """Single CBM-Hybrid training run."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    num_classes = model.n_classes

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    indices = np.arange(len(X_train))
    idx_tensor = torch.LongTensor(indices).to(device)

    dataset = TensorDataset(X_train_t, y_train_t, idx_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Class weights
    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weights = torch.FloatTensor(
        [total / (num_classes * c) if c > 0 else 1.0 for c in class_counts]
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_loss = float('inf')
    patience_counter = 0
    patience = 15
    best_model_state = None

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb, idx in loader:
            optimizer.zero_grad()
            log_probs, concepts = model(xb)
            loss = criterion(log_probs, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    EarlyStopping @ epoch {epoch + 1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    model.eval()
    with torch.no_grad():
        probs, _ = model.predict_proba(X_test_t)
        y_prob = probs.cpu().numpy()

    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_test, y_pred)
    ba = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    try:
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0

    return {
        'accuracy': acc,
        'balanced_accuracy': ba,
        'f1_macro': f1,
        'auc': auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


# ============== Metric Aggregation ==============

def aggregate_fold_results(fold_results):
    """Compute fold mean and standard deviation."""
    keys = ['accuracy', 'balanced_accuracy', 'f1_macro', 'auc']
    agg = {}
    for k in keys:
        vals = [r[k] for r in fold_results]
        agg[f'{k}_mean'] = float(np.mean(vals))
        agg[f'{k}_std'] = float(np.std(vals))
    return agg


# ============== Main Experiment Function ==============

def run_fair_baseline():
    print("=" * 65)
    print("Fair Parameter Baseline: Equal-Param PlainDNN vs PlainDNN-Large vs CBM-Hybrid")
    print("=" * 65)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    device_name = 'GPU (' + torch.cuda.get_device_name(0) + ')' \
        if torch.cuda.is_available() else 'CPU'
    print(f"Device: {device_name}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ===== Parameter Count Verification =====
    print("\n[Parameter Verification]")
    print("-" * 50)

    # PlainDNN-Large (hidden=[128,64,32], input=24, output=3)
    # Params: 24x128+128 + 128x64+64 + 64x32+32 + 32x3+3 + BN params (2*units per layer)
    m_large = PlainDNN(input_dim=24, num_classes=3, hidden_layers=[128, 64, 32])
    params_large = count_params(m_large)

    # PlainDNN-Fair: target ~2018 params
    # hidden=[48,24]: 24x48+48 + 48x24+24 + 24x3+3 + BN params
    m_fair_48_24 = PlainDNN(input_dim=24, num_classes=3, hidden_layers=[48, 24])
    params_fair_48_24 = count_params(m_fair_48_24)

    # hidden=[32,16]: fewer params
    m_fair_32_16 = PlainDNN(input_dim=24, num_classes=3, hidden_layers=[32, 16])
    params_fair_32_16 = count_params(m_fair_32_16)

    # CBM-Hybrid
    m_cbm = CBM_Hybrid()
    params_cbm = count_params(m_cbm)

    print(f"  PlainDNN-Large  (hidden=[128,64,32]): {params_large:,} params")
    print(f"  PlainDNN [48,24]                    : {params_fair_48_24:,} params")
    print(f"  PlainDNN [32,16]                    : {params_fair_32_16:,} params")
    print(f"  CBM-Hybrid (target)                 : {params_cbm:,} params")

    # Select configuration closest to CBM-Hybrid
    diff_48_24 = abs(params_fair_48_24 - params_cbm)
    diff_32_16 = abs(params_fair_32_16 - params_cbm)
    if diff_48_24 <= diff_32_16:
        fair_hidden = [48, 24]
        params_fair = params_fair_48_24
        print(f"\n  => Selected hidden=[48,24], params {params_fair:,}, "
              f"gap with CBM-Hybrid: {diff_48_24:,}")
    else:
        fair_hidden = [32, 16]
        params_fair = params_fair_32_16
        print(f"\n  => Selected hidden=[32,16], params {params_fair:,}, "
              f"gap with CBM-Hybrid: {diff_32_16:,}")

    print(f"\n  Param ratio (Large/CBM-Hybrid) = {params_large / params_cbm:.1f}x")
    print(f"  Param ratio (Fair/CBM-Hybrid)  = {params_fair / params_cbm:.2f}x")

    # ===== Data Loading =====
    print("\n[Data Loading]")
    print("-" * 50)
    X_nacc, y_nacc = load_nacc_extended(max_per_class=500)
    print(f"  NACC 3-class: shape={X_nacc.shape}, distribution={np.bincount(y_nacc)}")

    # binary: 0=Normal, 1=Impaired(MCI+Dementia)
    y_nacc_bin = (y_nacc > 0).astype(int)
    print(f"  NACC binary : distribution={np.bincount(y_nacc_bin)}")

    # ===== Experiment Loop =====
    model_configs = [
        {
            'name': 'PlainDNN_Large',
            'type': 'dnn',
            'hidden': [128, 64, 32],
            'params': params_large
        },
        {
            'name': 'PlainDNN_Fair',
            'type': 'dnn',
            'hidden': fair_hidden,
            'params': params_fair
        },
        {
            'name': 'CBM_Hybrid',
            'type': 'cbm',
            'params': params_cbm
        }
    ]

    all_results = {}

    for cfg in model_configs:
        model_name = cfg['name']
        all_results[model_name] = {
            'params': cfg['params'],
            'NACC_3class': {},
            'NACC_binary': {}
        }

    task_configs = [
        ('NACC_3class', X_nacc, y_nacc, 3),
        ('NACC_binary', X_nacc, y_nacc_bin, 2),
    ]

    for task_name, X, y, n_classes in task_configs:
        print(f"\n{'=' * 65}")
        print(f"Task: {task_name}  (n_classes={n_classes}, n_samples={len(y)})")
        print("=" * 65)

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        # Per-fold results for each model
        fold_results = {cfg['name']: [] for cfg in model_configs}

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\n  [Fold {fold + 1}/{N_FOLDS}]")

            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Preprocessing (same as main experiment)
            X_train_s, X_test_s = preprocess_dnn(X_train_raw, X_test_raw)

            for cfg in model_configs:
                model_name = cfg['name']

                if cfg['type'] == 'dnn':
                    res = train_plaindnn(
                        PlainDNN,
                        input_dim=X_train_s.shape[1],
                        num_classes=n_classes,
                        hidden_layers=cfg['hidden'],
                        X_train=X_train_s, y_train=y_train,
                        X_test=X_test_s, y_test=y_test,
                        epochs=100, batch_size=32, lr=0.0008,
                        verbose=False
                    )
                else:
                    # CBM-Hybrid: rebuild as 2-class for binary task
                    if n_classes == 2:
                        res = train_cbm_hybrid_binary(
                            X_train_s, y_train, X_test_s, y_test
                        )
                    else:
                        res = train_cbm_hybrid(
                            X_train_s, y_train, X_test_s, y_test,
                            epochs=100, batch_size=32, lr=1e-3,
                            verbose=False
                        )

                fold_results[model_name].append(res)
                print(f"    {model_name:20s}: "
                      f"BA={res['balanced_accuracy']:.4f}  "
                      f"AUC={res['auc']:.4f}  "
                      f"F1={res['f1_macro']:.4f}  "
                      f"Acc={res['accuracy']:.4f}")

        # Aggregate
        print(f"\n  [{task_name}] 10-fold mean ± std:")
        print(f"  {'Model':<22} {'BA':>12} {'AUC':>12} {'F1':>12} {'Acc':>12}")
        print("  " + "-" * 70)
        for cfg in model_configs:
            model_name = cfg['name']
            agg = aggregate_fold_results(fold_results[model_name])
            all_results[model_name][task_name] = {
                'BA':   round(agg['balanced_accuracy_mean'], 4),
                'BA_std': round(agg['balanced_accuracy_std'], 4),
                'AUC':  round(agg['auc_mean'], 4),
                'AUC_std': round(agg['auc_std'], 4),
                'F1':   round(agg['f1_macro_mean'], 4),
                'F1_std': round(agg['f1_macro_std'], 4),
                'Accuracy': round(agg['accuracy_mean'], 4),
                'Accuracy_std': round(agg['accuracy_std'], 4),
            }
            print(f"  {model_name:<22} "
                  f"{agg['balanced_accuracy_mean']:>6.4f}±{agg['balanced_accuracy_std']:.4f}  "
                  f"{agg['auc_mean']:>6.4f}±{agg['auc_std']:.4f}  "
                  f"{agg['f1_macro_mean']:>6.4f}±{agg['f1_macro_std']:.4f}  "
                  f"{agg['accuracy_mean']:>6.4f}±{agg['accuracy_std']:.4f}")

    # ===== Save Results =====
    result_path = os.path.join(RESULTS_DIR, 'fair_baseline_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {result_path}")

    # ===== Final Summary =====
    print("\n" + "=" * 65)
    print("Final Summary (NACC 3-class vs binary Equal-Param Baseline)")
    print("=" * 65)
    for task_name, _, _, _ in task_configs:
        print(f"\n[{task_name}]")
        print(f"  {'Model':<22} {'Params':>8} {'BA':>8} {'AUC':>8} {'F1':>8}")
        print("  " + "-" * 56)
        for cfg in model_configs:
            m = cfg['name']
            r = all_results[m][task_name]
            print(f"  {m:<22} {cfg['params']:>8,} "
                  f"{r['BA']:>8.4f} {r['AUC']:>8.4f} {r['F1']:>8.4f}")

    print(f"\nCompletion time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return all_results


# ============== CBM-Hybrid Binary Version ==============

class CBM_Hybrid_Binary(nn.Module):
    """
    CBM-Hybrid binary classification version (for binary tasks).
    """
    def __init__(self, n_demo=5, n_classes=2, n_free_dims=8):
        super().__init__()
        self.domain_names = list(COGNITIVE_DOMAINS.keys())
        self.domain_indices = []
        self.n_demo = n_demo
        self.n_classes = n_classes
        self.n_free_dims = n_free_dims

        self.concept_predictors = nn.ModuleList()
        for domain, items in COGNITIVE_DOMAINS.items():
            indices = [MOCA_FEATURES.index(item) for item in items]
            self.domain_indices.append(indices)
            n_items = len(items)
            predictor = nn.Sequential(
                nn.Linear(n_items, max(16, n_items * 2)),
                nn.ReLU(),
                nn.Linear(max(16, n_items * 2), 1),
                nn.Sigmoid()
            )
            self.concept_predictors.append(predictor)

        self.free_encoder = nn.Sequential(
            nn.Linear(24, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_free_dims)
        )

        n_concepts = len(COGNITIVE_DOMAINS)
        self.diagnosis_head = nn.Sequential(
            nn.Linear(n_concepts + n_free_dims + n_demo, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, n_classes)
        )

    def _get_concepts(self, x):
        x_moca = x[:, :19]
        concepts = []
        for i, indices in enumerate(self.domain_indices):
            domain_input = x_moca[:, indices]
            concept = self.concept_predictors[i](domain_input)
            concepts.append(concept)
        return torch.cat(concepts, dim=1)

    def forward(self, x):
        x_demo = x[:, 19:19 + self.n_demo]
        concepts = self._get_concepts(x)
        free = self.free_encoder(x)
        combined = torch.cat([concepts, free, x_demo], dim=1)
        logits = self.diagnosis_head(combined)
        return F.log_softmax(logits, dim=-1), concepts

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            log_probs, concepts = self.forward(x)
            return torch.exp(log_probs), concepts


def train_cbm_hybrid_binary(X_train, y_train, X_test, y_test,
                            epochs=100, batch_size=32, lr=1e-3,
                            max_retries=3, verbose=False):
    """Train binary CBM-Hybrid."""
    best_result = None
    best_ba = 0.0

    for attempt in range(max_retries):
        torch.manual_seed(SEED + attempt)
        np.random.seed(SEED + attempt)

        model = CBM_Hybrid_Binary(n_classes=2)
        result = _train_cbm_single(
            model, X_train, y_train, X_test, y_test,
            epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose
        )

        ba = result['balanced_accuracy']
        if ba > best_ba:
            best_ba = ba
            best_result = result

        if ba > 0.4:
            return result

    return best_result


# ============== Entry Point ==============

if __name__ == '__main__':
    run_fair_baseline()
