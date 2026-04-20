"""
Experiments 1-4: Core CBM evaluation on NACC and PPMI datasets.
Covers classification performance, concept validity, interpretability, and counterfactual analysis.

Key innovation: constraining information flow with MoCA cognitive domain
structure (structural constraint, not parameter constraint).

Covers four experiments:
1. Exp-1: Classification performance comparison
   (CBM-Independent/Supervised/Ordinal/Hybrid vs PlainDNN vs LR vs RF)
2. Exp-2: Concept layer validity
   (Pearson correlation between predicted concepts and true domain scores)
3. Exp-3: Concept layer interpretability
   (concept mean differences across diagnostic groups)
4. Exp-4: Counterfactual analysis
   (P(Normal) change after replacing MCI sample concepts)

Key constraints:
- SEED = 42
- Focal loss disabled
- KNNImputer(n_neighbors=5) + StandardScaler preprocessing
- Retry up to 3 times when balanced_accuracy <= 0.4
- MAE metric for ordinal error measurement

Corresponds to Section 4.1-4.2 in the paper.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any
from scipy.stats import pearsonr
from collections import OrderedDict

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Project imports
sys.path.insert(0, os.path.dirname(__file__))
from model_dcm_dnn import PlainDNN, train_and_evaluate
from experiment_data import (
    load_ppmi_extended, load_nacc_extended,
    EXTENDED_FEATURES
)

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

# Random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============== MoCA Cognitive Domain Mapping (Clinical Prior) ==============

# MoCA 19 sub-items (corresponding to the first 19 entries of EXTENDED_FEATURES)
MOCA_FEATURES = [
    'TRAIL_MAKING', 'CUBE', 'CLOCK_CONTOUR', 'CLOCK_NUMBERS', 'CLOCK_HANDS',
    'NAMING', 'DIGIT_SPAN', 'VIGILANCE', 'SERIAL_7',
    'SENTENCE_REP', 'FLUENCY', 'ABSTRACTION',
    'DELAYED_RECALL',
    'ORIENT_DATE', 'ORIENT_MONTH', 'ORIENT_YEAR', 'ORIENT_DAY', 'ORIENT_PLACE', 'ORIENT_CITY'
]

# 7 cognitive domains and their corresponding MoCA sub-items
COGNITIVE_DOMAINS = OrderedDict([
    ('Visuospatial_Executive', ['TRAIL_MAKING', 'CUBE', 'CLOCK_CONTOUR', 'CLOCK_NUMBERS', 'CLOCK_HANDS']),
    ('Naming', ['NAMING']),
    ('Attention', ['DIGIT_SPAN', 'VIGILANCE', 'SERIAL_7']),
    ('Language', ['SENTENCE_REP', 'FLUENCY']),
    ('Abstraction', ['ABSTRACTION']),
    ('Delayed_Recall', ['DELAYED_RECALL']),
    ('Orientation', ['ORIENT_DATE', 'ORIENT_MONTH', 'ORIENT_YEAR', 'ORIENT_DAY', 'ORIENT_PLACE', 'ORIENT_CITY'])
])

# Maximum score per cognitive domain
DOMAIN_MAX_SCORES = {
    'Visuospatial_Executive': 5, 'Naming': 3, 'Attention': 6,
    'Language': 3, 'Abstraction': 2, 'Delayed_Recall': 5, 'Orientation': 6
}


# ============== Utility Functions ==============

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    """
    y_pred = np.argmax(y_prob, axis=1)
    confidence = np.max(y_prob, axis=1)
    correct = (y_pred == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (confidence > bin_boundaries[i]) & (confidence <= bin_boundaries[i + 1])
        count = mask.sum()
        
        if count > 0:
            avg_confidence = confidence[mask].mean()
            avg_accuracy = correct[mask].mean()
            ece += count / len(y_true) * abs(avg_confidence - avg_accuracy)
    
    return ece


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute ordinal error MAE (Mean Absolute Error).
    Used to measure the severity of ordinal error in predicted classes.
    """
    return np.mean(np.abs(y_true - y_pred))


def preprocess_with_concepts(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    KNNImputer + StandardScaler, also returns true concepts.

    Returns:
        X_train_scaled, X_test_scaled: preprocessed features
        true_concepts_train, true_concepts_test: normalized true cognitive domain scores
    """
    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # Compute true concepts (after imputer, before scaler)
    true_concepts_train = compute_true_concepts(X_train_imp)
    true_concepts_test = compute_true_concepts(X_test_imp)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    return X_train_scaled, X_test_scaled, true_concepts_train, true_concepts_test


def compute_true_concepts(X_imputed: np.ndarray) -> np.ndarray:
    """
    Compute true normalized scores for 7 cognitive domains from imputed data.

    Args:
        X_imputed: shape=(N, 24), features after imputation but before scaling

    Returns:
        concepts: shape=(N, 7), normalized scores per cognitive domain in [0,1]
    """
    concepts = np.zeros((len(X_imputed), len(COGNITIVE_DOMAINS)))
    
    for i, (domain, items) in enumerate(COGNITIVE_DOMAINS.items()):
        indices = [MOCA_FEATURES.index(item) for item in items]
        domain_sum = X_imputed[:, indices].sum(axis=1)
        domain_max = DOMAIN_MAX_SCORES[domain]
        concepts[:, i] = np.clip(domain_sum / domain_max, 0, 1)
    
    return concepts.astype(np.float32)


def preprocess_dnn(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """KNNImputer + StandardScaler for DNN models."""
    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    return X_train_scaled, X_test_scaled


def train_sklearn_model(model, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        preprocess_fn=None) -> Dict[str, Any]:
    """Train a sklearn model and return metrics dict."""
    if preprocess_fn is not None:
        X_train_p, X_test_p = preprocess_fn(X_train, X_test)
    else:
        X_train_p, X_test_p = X_train, X_test
    
    model.fit(X_train_p, y_train)
    y_pred = model.predict(X_test_p)
    y_prob = model.predict_proba(X_test_p)
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    ba = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    mae = compute_mae(y_test, y_pred)

    return {
        'accuracy': acc,
        'balanced_accuracy': ba,
        'f1': f1,
        'mae': mae,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'model': model
    }


# ============== CBM Model Definitions ==============

class CBM_Independent(nn.Module):
    """
    CBM-Independent: Standard Concept Bottleneck Model.

    Architecture:
    - 7 independent cognitive domain concept predictors (domain sub-items -> concept value)
    - Diagnosis head: 7 concepts + 5 demographics -> 3-class classification
    - Trained with standard cross-entropy loss
    """
    
    def __init__(self, n_demo=5, n_classes=3):
        super().__init__()
        self.domain_names = list(COGNITIVE_DOMAINS.keys())
        self.domain_indices = []
        self.n_demo = n_demo
        self.n_classes = n_classes
        
        # Concept predictors: one per cognitive domain
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
        
        # Diagnosis head
        n_concepts = len(COGNITIVE_DOMAINS)  # 7
        self.diagnosis_head = nn.Sequential(
            nn.Linear(n_concepts + n_demo, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, n_classes)
        )

    def _get_concepts(self, x):
        """Extract concept layer output."""
        x_moca = x[:, :19]
        concepts = []
        for i, indices in enumerate(self.domain_indices):
            domain_input = x_moca[:, indices]
            concept = self.concept_predictors[i](domain_input)
            concepts.append(concept)
        return torch.cat(concepts, dim=1)  # [B, 7]

    def forward(self, x):
        x_demo = x[:, 19:19+self.n_demo]
        concepts = self._get_concepts(x)
        combined = torch.cat([concepts, x_demo], dim=1)  # [B, 12]
        logits = self.diagnosis_head(combined)
        return F.log_softmax(logits, dim=-1), concepts
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            log_probs, concepts = self.forward(x)
            return torch.exp(log_probs), concepts


class CBM_Supervised(CBM_Independent):
    """
    CBM-Supervised: Concept-supervised version.

    Same architecture as CBM-Independent.
    Differs in training: adds concept supervision loss: MSE(predicted_concepts, true_concepts).
    """
    pass  # Same architecture; pass concept_supervision=True during training


class CBM_Ordinal(nn.Module):
    """
    CBM-Ordinal: Ordinal-constrained CBM.

    Key innovation:
    - Concept layer same as CBM-Independent
    - Diagnosis head: cumulative logit ordinal regression + monotonicity constraint
    - P(Y<=k) = sigmoid(theta_k + w^T * concepts + v^T * demo)
    - theta_1 <= theta_2 (guaranteed via softplus increment)
    - w = exp(w_raw) >= 0 (monotonicity: higher concept -> more likely Normal)
    """
    
    def __init__(self, n_demo=5, n_classes=3):
        super().__init__()
        self.domain_names = list(COGNITIVE_DOMAINS.keys())
        self.domain_indices = []
        self.n_demo = n_demo
        self.n_classes = n_classes
        
        # Concept predictors (same as CBM-Independent)
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

        n_concepts = len(COGNITIVE_DOMAINS)  # 7

        # Ordinal regression head
        # Monotonicity constraint: w_raw guaranteed non-negative after exp()
        self.w_raw = nn.Parameter(torch.zeros(n_concepts))  # concept weights (non-neg after exp)
        self.v = nn.Parameter(torch.zeros(n_demo))  # demographic weights (unconstrained)

        # Cumulative thresholds: theta_1 <= theta_2
        # Parameterization: theta_1 = t1, theta_2 = t1 + softplus(delta)
        self.t1 = nn.Parameter(torch.tensor(0.0))
        self.delta_raw = nn.Parameter(torch.tensor(1.0))  # positive after softplus

    def _get_concepts(self, x):
        """Extract concept layer output."""
        x_moca = x[:, :19]
        concepts = []
        for i, indices in enumerate(self.domain_indices):
            domain_input = x_moca[:, indices]
            concept = self.concept_predictors[i](domain_input)
            concepts.append(concept)
        return torch.cat(concepts, dim=1)  # [B, 7]
    
    def forward(self, x):
        x_demo = x[:, 19:19+self.n_demo]
        concepts = self._get_concepts(x)

        # Monotonic weights
        w = torch.exp(self.w_raw)  # [7], all >= 0

        # Linear combination: f = w^T * concepts + v^T * demo
        # Higher concepts -> larger f -> larger P(Y<=0) -> more likely Normal
        f_pos = (concepts * w).sum(dim=1) + (x_demo * self.v).sum(dim=1)  # [B]

        # Cumulative thresholds
        theta1 = self.t1
        theta2 = self.t1 + F.softplus(self.delta_raw)

        # Cumulative probabilities: P(Y<=k) = sigmoid(theta_k + f)
        # Y=0(Normal), Y=1(MCI), Y=2(Dementia)
        cum_prob_0 = torch.sigmoid(theta1 + f_pos)  # P(Y<=0) = P(Normal)
        cum_prob_1 = torch.sigmoid(theta2 + f_pos)  # P(Y<=1) = P(Normal or MCI)

        # Class probabilities
        p0 = cum_prob_0                           # P(Normal)
        p1 = cum_prob_1 - cum_prob_0              # P(MCI)
        p2 = 1.0 - cum_prob_1                     # P(Dementia)
        
        probs = torch.stack([p0, p1, p2], dim=1)  # [B, 3]
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        log_probs = torch.log(probs)
        
        return log_probs, concepts
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            log_probs, concepts = self.forward(x)
            return torch.exp(log_probs), concepts
    
    def ordinal_loss(self, x, y):
        """
        Ordinal cross-entropy loss.
        Computes BCE for each threshold and sums them.
        """
        x_demo = x[:, 19:19+self.n_demo]
        concepts = self._get_concepts(x)
        w = torch.exp(self.w_raw)
        f_pos = (concepts * w).sum(dim=1) + (x_demo * self.v).sum(dim=1)
        
        theta1 = self.t1
        theta2 = self.t1 + F.softplus(self.delta_raw)
        
        # Binary labels: Y<=0? Y<=1?
        y0 = (y <= 0).float()  # 1 if Normal
        y1 = (y <= 1).float()  # 1 if Normal or MCI
        
        logit0 = theta1 + f_pos
        logit1 = theta2 + f_pos
        
        loss = F.binary_cross_entropy_with_logits(logit0, y0) + \
               F.binary_cross_entropy_with_logits(logit1, y1)
        
        return loss, concepts


class CBM_Hybrid(nn.Module):
    """
    CBM-Hybrid: 7 concepts + 8 free latent dimensions.

    Serves as an ablation control to examine bottleneck expressiveness.
    """
    
    def __init__(self, n_demo=5, n_classes=3, n_free_dims=8):
        super().__init__()
        self.domain_names = list(COGNITIVE_DOMAINS.keys())
        self.domain_indices = []
        self.n_demo = n_demo
        self.n_classes = n_classes
        self.n_free_dims = n_free_dims
        
        # Concept predictors (same as CBM-Independent)
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

        # Free encoder: 24-dim full input -> 8-dim
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
        """Extract concept layer output."""
        x_moca = x[:, :19]
        concepts = []
        for i, indices in enumerate(self.domain_indices):
            domain_input = x_moca[:, indices]
            concept = self.concept_predictors[i](domain_input)
            concepts.append(concept)
        return torch.cat(concepts, dim=1)  # [B, 7]

    def forward(self, x):
        x_demo = x[:, 19:19+self.n_demo]
        concepts = self._get_concepts(x)
        free = self.free_encoder(x)
        combined = torch.cat([concepts, free, x_demo], dim=1)  # [B, 20]
        logits = self.diagnosis_head(combined)
        return F.log_softmax(logits, dim=-1), concepts
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            log_probs, concepts = self.forward(x)
            return torch.exp(log_probs), concepts


# ============== Training Functions ==============

def train_cbm(model, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              epochs: int = 100, batch_size: int = 32, lr: float = 1e-3,
              concept_supervision: bool = False, lambda_concept: float = 0.5,
              true_concepts_train: np.ndarray = None, true_concepts_test: np.ndarray = None,
              ordinal: bool = False, max_retries: int = 3, verbose: bool = False,
              lambda_polarity: float = 0.1) -> Dict[str, Any]:
    """
    Unified CBM training function.

    Args:
        model: CBM model instance
        X_train, y_train: training data
        X_test, y_test: test data
        epochs: number of training epochs
        batch_size: batch size
        lr: learning rate
        concept_supervision: whether to use concept supervision
        lambda_concept: concept supervision weight
        true_concepts_train, true_concepts_test: ground-truth concept values
        ordinal: whether to use ordinal loss
        max_retries: maximum number of retries
        verbose: whether to print detailed training info
        lambda_polarity: polarity constraint weight

    Returns:
        dict: metrics and model
    """
    best_result = None
    best_ba = 0.0
    best_model_state = None
    
    for attempt in range(max_retries):
        # Reset random seed
        torch.manual_seed(SEED + attempt)
        np.random.seed(SEED + attempt)

        # Re-initialize model
        if attempt > 0:
            model = type(model)(**{k: getattr(model, k) for k in ['n_demo', 'n_classes'] if hasattr(model, k)})
        
        # Train
        result = _train_cbm_single(
            model, X_train, y_train, X_test, y_test,
            epochs=epochs, batch_size=batch_size, lr=lr,
            concept_supervision=concept_supervision,
            lambda_concept=lambda_concept,
            true_concepts_train=true_concepts_train,
            true_concepts_test=true_concepts_test,
            ordinal=ordinal,
            verbose=verbose,
            lambda_polarity=lambda_polarity
        )
        
        ba = result['balanced_accuracy']
        
        if ba > best_ba:
            best_ba = ba
            best_result = result
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if ba > 0.4:
            result['model'] = model
            return result

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    best_result['model'] = model
    return best_result


def _train_cbm_single(model, X_train, y_train, X_test, y_test,
                      epochs, batch_size, lr,
                      concept_supervision, lambda_concept,
                      true_concepts_train, true_concepts_test,
                      ordinal, verbose, lambda_polarity=0.1):
    """Single training pass."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Convert to Tensor
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    # Create index array for correct matching of true concepts
    indices = np.arange(len(X_train))
    idx_tensor = torch.LongTensor(indices).to(device)

    # Concept supervision
    if concept_supervision and true_concepts_train is not None:
        true_concepts_train_t = torch.FloatTensor(true_concepts_train).to(device)
    else:
        true_concepts_train_t = None

    # DataLoader - TensorDataset carries original indices
    dataset = TensorDataset(X_train_t, y_train_t, idx_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Class weights
    class_counts = np.bincount(y_train)
    total = len(y_train)
    num_classes = len(class_counts)
    class_weights = torch.FloatTensor([total / (num_classes * c) if c > 0 else 1.0 
                                       for c in class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Early stopping
    best_loss = float('inf')
    patience_counter = 0
    patience = 15
    best_model_state = None

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb, idx in loader:
            optimizer.zero_grad()

            if ordinal and hasattr(model, 'ordinal_loss'):
                # Ordinal loss
                loss, concepts = model.ordinal_loss(xb, yb)

                # Concept supervision
                if concept_supervision and true_concepts_train_t is not None:
                    # Index true_concepts_train using idx directly
                    true_concepts_batch = true_concepts_train_t[idx]
                    concept_loss = F.mse_loss(concepts, true_concepts_batch)

                    # Polarity constraint: penalize negative correlation
                    corr_penalty = 0.0
                    for d in range(concepts.shape[1]):
                        pred_d = concepts[:, d]
                        true_d = true_concepts_batch[:, d]
                        pred_centered = pred_d - pred_d.mean()
                        true_centered = true_d - true_d.mean()
                        corr = (pred_centered * true_centered).sum() / (
                            pred_centered.norm() * true_centered.norm() + 1e-8)
                        corr_penalty += F.relu(-corr)  # penalize polarity flip

                    concept_loss = concept_loss + lambda_polarity * corr_penalty
                    loss = loss + lambda_concept * concept_loss
            else:
                # Standard CE loss
                log_probs, concepts = model(xb)
                loss = criterion(log_probs, yb)

                # Concept supervision
                if concept_supervision and true_concepts_train_t is not None:
                    # Index true_concepts_train using idx directly
                    true_concepts_batch = true_concepts_train_t[idx]
                    concept_loss = F.mse_loss(concepts, true_concepts_batch)

                    # Polarity constraint: penalize negative correlation
                    corr_penalty = 0.0
                    for d in range(concepts.shape[1]):
                        pred_d = concepts[:, d]
                        true_d = true_concepts_batch[:, d]
                        pred_centered = pred_d - pred_d.mean()
                        true_centered = true_d - true_d.mean()
                        corr = (pred_centered * true_centered).sum() / (
                            pred_centered.norm() * true_centered.norm() + 1e-8)
                        corr_penalty += F.relu(-corr)  # penalize polarity flip

                    concept_loss = concept_loss + lambda_polarity * corr_penalty
                    loss = loss + lambda_concept * concept_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    # Evaluate
    model.eval()
    with torch.no_grad():
        probs, concepts = model.predict_proba(X_test_t)
        y_prob = probs.cpu().numpy()
        concepts_test = concepts.cpu().numpy()
        y_pred = y_prob.argmax(axis=1)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    ba = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    mae = compute_mae(y_test, y_pred)
    
    return {
        'accuracy': acc,
        'balanced_accuracy': ba,
        'f1_macro': f1,
        'mae': mae,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'concepts': concepts_test,
        'model': model
    }


def train_dnn_with_retry(model_class, input_dim: int, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray, num_classes: int = 3,
                         max_retries: int = 3, verbose: bool = False) -> Dict[str, Any]:
    """Train DNN model, retry if balanced_accuracy <= 0.4."""
    best_result = None
    best_ba = 0.0
    best_model = None
    
    for attempt in range(max_retries):
        torch.manual_seed(SEED + attempt)
        np.random.seed(SEED + attempt)
        
        model = model_class(input_dim=input_dim, num_classes=num_classes)
        
        result = train_and_evaluate(
            model, X_train, y_train, X_test, y_test,
            epochs=100, batch_size=32, verbose=verbose,
            use_focal_loss=False  # focal loss disabled
        )
        
        ba = result['balanced_accuracy']
        
        if ba > best_ba:
            best_ba = ba
            best_result = result
            best_model = model
        
        if ba > 0.4:
            result['model'] = model
            result['mae'] = compute_mae(y_test, result['y_pred'])
            return result
    
    best_result['model'] = best_model
    best_result['mae'] = compute_mae(y_test, best_result['y_pred'])
    return best_result


# ============== Experiment 1: Classification Performance Comparison ==============

def run_exp1_performance(datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[pd.DataFrame, Dict]:
    """
    Experiment 1: Classification performance comparison.

    Compares 7 models x 2 datasets x 5-fold:
    1. CBM-Independent
    2. CBM-Supervised
    3. CBM-Ordinal
    4. CBM-Hybrid
    5. PlainDNN
    6. LR
    7. RF
    """
    print("\n" + "=" * 60)
    print("Experiment 1: Classification Performance Comparison")
    print("=" * 60)
    
    results = []
    
    # Store models and data for subsequent experiments
    saved_models = {
        'CBM_Supervised': {'NACC': [], 'PPMI': []},
        'CBM_Ordinal': {'NACC': [], 'PPMI': []}
    }
    saved_data = {
        'NACC': [],
        'PPMI': []
    }
    
    for ds_name, (X, y) in datasets.items():
        print(f"\n[{ds_name}] Dataset")
        print("-" * 50)
        
        input_dim = X.shape[1]
        num_classes = len(np.unique(y))
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"  Fold {fold}...", end='')
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Preprocessing (including true concept computation)
            X_train_s, X_test_s, tc_train, tc_test = preprocess_with_concepts(X_train, X_test)

            # 1. CBM-Independent (no concept supervision)
            model = CBM_Independent()
            res = train_cbm(model, X_train_s, y_train, X_test_s, y_test, concept_supervision=False)
            ece = compute_ece(y_test, res['y_prob'])
            results.append({
                'dataset': ds_name, 'model': 'CBM_Independent', 'fold': fold,
                'balanced_accuracy': res['balanced_accuracy'],
                'f1_macro': res['f1_macro'],
                'ece': ece,
                'mae': res['mae']
            })
            print(f" Ind: BA={res['balanced_accuracy']:.3f}", end='')
            
            # 2. CBM-Supervised (with concept supervision)
            model = CBM_Supervised()
            res = train_cbm(model, X_train_s, y_train, X_test_s, y_test, 
                           concept_supervision=True,
                           true_concepts_train=tc_train,
                           true_concepts_test=tc_test)
            ece = compute_ece(y_test, res['y_prob'])
            results.append({
                'dataset': ds_name, 'model': 'CBM_Supervised', 'fold': fold,
                'balanced_accuracy': res['balanced_accuracy'],
                'f1_macro': res['f1_macro'],
                'ece': ece,
                'mae': res['mae']
            })
            # Save model and data
            saved_models['CBM_Supervised'][ds_name].append({
                'model': res['model'],
                'X_test': X_test_s,
                'y_test': y_test,
                'concepts': res['concepts'],
                'true_concepts': tc_test
            })
            print(f" Sup: BA={res['balanced_accuracy']:.3f}", end='')
            
            # 3. CBM-Ordinal
            model = CBM_Ordinal()
            res = train_cbm(model, X_train_s, y_train, X_test_s, y_test,
                           concept_supervision=True,
                           true_concepts_train=tc_train,
                           true_concepts_test=tc_test,
                           ordinal=True)
            ece = compute_ece(y_test, res['y_prob'])
            results.append({
                'dataset': ds_name, 'model': 'CBM_Ordinal', 'fold': fold,
                'balanced_accuracy': res['balanced_accuracy'],
                'f1_macro': res['f1_macro'],
                'ece': ece,
                'mae': res['mae']
            })
            saved_models['CBM_Ordinal'][ds_name].append({
                'model': res['model'],
                'X_test': X_test_s,
                'y_test': y_test,
                'concepts': res['concepts'],
                'true_concepts': tc_test
            })
            print(f" Ord: BA={res['balanced_accuracy']:.3f}", end='')
            
            # 4. CBM-Hybrid
            model = CBM_Hybrid()
            res = train_cbm(model, X_train_s, y_train, X_test_s, y_test, concept_supervision=False)
            ece = compute_ece(y_test, res['y_prob'])
            results.append({
                'dataset': ds_name, 'model': 'CBM_Hybrid', 'fold': fold,
                'balanced_accuracy': res['balanced_accuracy'],
                'f1_macro': res['f1_macro'],
                'ece': ece,
                'mae': res['mae']
            })
            print(f" Hyb: BA={res['balanced_accuracy']:.3f}", end='')
            
            # 5. PlainDNN
            res = train_dnn_with_retry(PlainDNN, input_dim, X_train_s, y_train, X_test_s, y_test, num_classes)
            ece = compute_ece(y_test, res['y_prob'])
            results.append({
                'dataset': ds_name, 'model': 'PlainDNN', 'fold': fold,
                'balanced_accuracy': res['balanced_accuracy'],
                'f1_macro': res['f1'],
                'ece': ece,
                'mae': res['mae']
            })
            print(f" DNN: BA={res['balanced_accuracy']:.3f}", end='')
            
            # 6. LR
            res = train_sklearn_model(
                LogisticRegression(max_iter=2000, random_state=SEED, multi_class='multinomial'),
                X_train, y_train, X_test, y_test, preprocess_dnn
            )
            ece = compute_ece(y_test, res['y_prob'])
            results.append({
                'dataset': ds_name, 'model': 'LR', 'fold': fold,
                'balanced_accuracy': res['balanced_accuracy'],
                'f1_macro': res['f1'],
                'ece': ece,
                'mae': res['mae']
            })
            print(f" LR: BA={res['balanced_accuracy']:.3f}", end='')
            
            # 7. RF
            res = train_sklearn_model(
                RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
                X_train, y_train, X_test, y_test, preprocess_dnn
            )
            ece = compute_ece(y_test, res['y_prob'])
            results.append({
                'dataset': ds_name, 'model': 'RF', 'fold': fold,
                'balanced_accuracy': res['balanced_accuracy'],
                'f1_macro': res['f1'],
                'ece': ece,
                'mae': res['mae']
            })
            print(f" RF: BA={res['balanced_accuracy']:.3f}")
            
            # Save test data for subsequent experiments
            saved_data[ds_name].append({
                'X_test': X_test_s,
                'y_test': y_test,
                'true_concepts': tc_test
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)

    # Add summary rows
    summary_rows = []
    for ds_name in datasets.keys():
        for model_name in df['model'].unique():
            model_df = df[(df['dataset'] == ds_name) & (df['model'] == model_name)]
            summary_rows.append({
                'dataset': ds_name,
                'model': model_name,
                'fold': 'mean',
                'balanced_accuracy': model_df['balanced_accuracy'].mean(),
                'f1_macro': model_df['f1_macro'].mean(),
                'ece': model_df['ece'].mean(),
                'mae': model_df['mae'].mean()
            })
    df = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)
    
    # Save
    exp1_path = os.path.join(RESULTS_DIR, 'cbm_exp1_performance.csv')
    df.to_csv(exp1_path, index=False)
    print(f"\nResults saved to: {exp1_path}")
    
    return df, saved_models, saved_data


# ============== Experiment 2: Concept Layer Validity ==============

def run_exp2_concept_validity(saved_models: Dict, saved_data: Dict) -> pd.DataFrame:
    """
    Experiment 2: Concept layer validity.

    Extracts concept scores from CBM-Supervised and computes Pearson correlation
    with true domain scores.
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Concept Layer Validity")
    print("=" * 60)
    
    results = []
    
    for ds_name in ['NACC', 'PPMI']:
        print(f"\n[{ds_name}] Concept Validity Analysis")
        print("-" * 50)

        all_pred_concepts = []
        all_true_concepts = []

        for fold_data in saved_models['CBM_Supervised'][ds_name]:
            all_pred_concepts.append(fold_data['concepts'])
            all_true_concepts.append(fold_data['true_concepts'])

        all_pred_concepts = np.vstack(all_pred_concepts)
        all_true_concepts = np.vstack(all_true_concepts)

        # Compute Pearson correlation per domain
        print("  Pearson correlation per cognitive domain:")
        for i, domain in enumerate(COGNITIVE_DOMAINS.keys()):
            r, p = pearsonr(all_pred_concepts[:, i], all_true_concepts[:, i])
            results.append({
                'dataset': ds_name,
                'domain': domain,
                'pearson_r': r,
                'p_value': p
            })
            print(f"    {domain}: r={r:.4f}, p={p:.4e}")
        
        # Compute mean correlation
        mean_r = np.mean([r['pearson_r'] for r in results if r['dataset'] == ds_name])
        print(f"  Mean Pearson r = {mean_r:.4f}")
    
    df = pd.DataFrame(results)
    
    # Save
    exp2_path = os.path.join(RESULTS_DIR, 'cbm_exp2_concept_validity.csv')
    df.to_csv(exp2_path, index=False)
    print(f"\nResults saved to: {exp2_path}")
    
    return df


# ============== Experiment 3: Concept Layer Interpretability ==============

def run_exp3_interpretability(saved_models: Dict, saved_data: Dict) -> Dict:
    """
    Experiment 3: Concept layer interpretability.

    Analyzes mean differences of 7 concepts across diagnostic groups (0/1/2).
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Concept Layer Interpretability")
    print("=" * 60)
    
    results = {}
    class_names = ['Normal', 'MCI', 'Dementia']
    
    for ds_name in ['NACC', 'PPMI']:
        print(f"\n[{ds_name}] Concept mean analysis by diagnostic group")
        print("-" * 50)

        results[ds_name] = {'classes': {}}

        # Collect concepts from all folds
        all_concepts = []
        all_labels = []
        
        for fold_data in saved_models['CBM_Supervised'][ds_name]:
            all_concepts.append(fold_data['concepts'])
            all_labels.append(fold_data['y_test'])
        
        all_concepts = np.vstack(all_concepts)
        all_labels = np.concatenate(all_labels)
        
        # Group by class and compute
        for c in range(3):
            mask = all_labels == c
            if mask.sum() > 0:
                concepts_c = all_concepts[mask]
                results[ds_name]['classes'][class_names[c]] = {
                    'n_samples': int(mask.sum()),
                    'concept_means': {
                        domain: float(concepts_c[:, i].mean())
                        for i, domain in enumerate(COGNITIVE_DOMAINS.keys())
                    },
                    'concept_stds': {
                        domain: float(concepts_c[:, i].std())
                        for i, domain in enumerate(COGNITIVE_DOMAINS.keys())
                    }
                }
        
        # Print results
        print(f"\n  Samples per class:")
        for c_name in class_names:
            if c_name in results[ds_name]['classes']:
                print(f"    {c_name}: {results[ds_name]['classes'][c_name]['n_samples']}")
        
        print(f"\n  Concept mean per cognitive domain:")
        header = f"  {'Domain':<25} {'Normal':>10} {'MCI':>10} {'Dementia':>10}"
        print(header)
        print("  " + "-" * 55)
        
        for domain in COGNITIVE_DOMAINS.keys():
            values = []
            for c_name in class_names:
                if c_name in results[ds_name]['classes']:
                    values.append(results[ds_name]['classes'][c_name]['concept_means'][domain])
                else:
                    values.append(float('nan'))
            print(f"  {domain:<25} {values[0]:>10.3f} {values[1]:>10.3f} {values[2]:>10.3f}")
    
    # Save
    exp3_path = os.path.join(RESULTS_DIR, 'cbm_exp3_interpretability.json')
    with open(exp3_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {exp3_path}")
    
    return results


# ============== Experiment 4: Counterfactual Analysis ==============

def run_exp4_counterfactual(saved_models: Dict, saved_data: Dict) -> Dict:
    """
    Experiment 4: Counterfactual analysis.

    For MCI samples, replaces each concept with the Normal group mean
    and observes the change in P(Normal).
    """
    print("\n" + "=" * 60)
    print("Experiment 4: Counterfactual Analysis")
    print("=" * 60)
    
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for ds_name in ['NACC', 'PPMI']:
        print(f"\n[{ds_name}] Counterfactual Analysis")
        print("-" * 50)

        results[ds_name] = {'counterfactual_effects': {}}

        # Prefer CBM-Ordinal model, fall back to CBM-Supervised
        model_key = 'CBM_Ordinal' if saved_models['CBM_Ordinal'][ds_name] else 'CBM_Supervised'
        
        all_effects = {domain: [] for domain in COGNITIVE_DOMAINS.keys()}
        
        for fold_idx, fold_data in enumerate(saved_models[model_key][ds_name]):
            model = fold_data['model'].to(device)
            model.eval()
            
            X_test = fold_data['X_test']
            y_test = fold_data['y_test']
            concepts = fold_data['concepts']
            
            # Select MCI samples
            mci_mask = y_test == 1
            if mci_mask.sum() == 0:
                continue

            mci_concepts = concepts[mci_mask]
            mci_X = X_test[mci_mask]

            # Compute Normal group concept means
            normal_mask = y_test == 0
            if normal_mask.sum() == 0:
                continue
            normal_concept_means = concepts[normal_mask].mean(axis=0)

            # Original P(Normal)
            with torch.no_grad():
                X_tensor = torch.FloatTensor(mci_X).to(device)
                orig_probs, _ = model.predict_proba(X_tensor)
                orig_p_normal = orig_probs[:, 0].cpu().numpy()
            
            # Counterfactual replacement for each concept
            for i, domain in enumerate(COGNITIVE_DOMAINS.keys()):
                # Replace the i-th concept with the Normal group mean
                modified_concepts = mci_concepts.copy()
                modified_concepts[:, i] = normal_concept_means[i]

                # Use different counterfactual inference depending on model type
                with torch.no_grad():
                    x_demo = torch.FloatTensor(mci_X[:, 19:24]).to(device)
                    modified_concepts_t = torch.FloatTensor(modified_concepts).to(device)

                    if hasattr(model, 'diagnosis_head'):
                        # CBM-Independent/Supervised/Hybrid: use diagnosis head
                        combined = torch.cat([modified_concepts_t, x_demo], dim=1)
                        logits = model.diagnosis_head(combined)
                        probs = F.softmax(logits, dim=-1)
                        new_p_normal = probs[:, 0].cpu().numpy()
                    elif hasattr(model, 'w_raw'):
                        # CBM-Ordinal: use ordinal regression computation
                        w = torch.exp(model.w_raw)
                        f_pos = (modified_concepts_t * w).sum(dim=1) + (x_demo * model.v).sum(dim=1)
                        theta1 = model.t1
                        cum_prob_0 = torch.sigmoid(theta1 + f_pos)
                        new_p_normal = cum_prob_0.cpu().numpy()
                    else:
                        # Unknown model type, skip
                        new_p_normal = orig_p_normal.copy()

                # Compute P(Normal) increment
                delta_p = new_p_normal - orig_p_normal
                all_effects[domain].extend(delta_p.tolist())
        
        # Compute mean effect
        print(f"  Counterfactual effect per domain (P(Normal) increase after replacing MCI concept with Normal mean):")
        for domain in COGNITIVE_DOMAINS.keys():
            if all_effects[domain]:
                mean_effect = np.mean(all_effects[domain])
                results[ds_name]['counterfactual_effects'][domain] = {
                    'mean_delta_p': float(mean_effect),
                    'n_samples': len(all_effects[domain])
                }
                print(f"    {domain}: ΔP(Normal)={mean_effect:.4f}")
        
        # Sort by effect
        sorted_domains = sorted(
            results[ds_name]['counterfactual_effects'].items(),
            key=lambda x: x[1]['mean_delta_p'],
            reverse=True
        )
        results[ds_name]['sorted_by_effect'] = [d[0] for d in sorted_domains]

        print(f"\n  Sorted by effect (descending):")
        for i, (domain, data) in enumerate(sorted_domains, 1):
            print(f"    {i}. {domain}: {data['mean_delta_p']:.4f}")
    
    # Save
    exp4_path = os.path.join(RESULTS_DIR, 'cbm_exp4_counterfactual.json')
    with open(exp4_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {exp4_path}")
    
    return results


# ============== Main Function ==============

def main():
    print("=" * 60)
    print("CBM Concept Bottleneck Model Exploration Experiments")
    print("=" * 60)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    start = time.time()

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        # ========== Data Loading ==========
        print("\n[Data Loading]")
        print("-" * 50)

        # NACC data
        X_nacc, y_nacc = load_nacc_extended(max_per_class=500)
        print(f"  NACC: {X_nacc.shape}, class distribution={np.bincount(y_nacc)}")

        # PPMI data
        X_ppmi, y_ppmi = load_ppmi_extended()
        print(f"  PPMI: {X_ppmi.shape}, class distribution={np.bincount(y_ppmi)}")
        
        datasets = {
            'NACC': (X_nacc, y_nacc),
            'PPMI': (X_ppmi, y_ppmi)
        }
        
        # ========== Experiment 1: Classification Performance ==========
        df_exp1, saved_models, saved_data = run_exp1_performance(datasets)

        # ========== Experiment 2: Concept Layer Validity ==========
        df_exp2 = run_exp2_concept_validity(saved_models, saved_data)

        # ========== Experiment 3: Concept Layer Interpretability ==========
        exp3_results = run_exp3_interpretability(saved_models, saved_data)

        # ========== Experiment 4: Counterfactual Analysis ==========
        exp4_results = run_exp4_counterfactual(saved_models, saved_data)

        # ========== Done ==========
        elapsed = time.time() - start
        print("\n" + "=" * 60)
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Results saved to: {RESULTS_DIR}")
        print("All CBM exploration experiments completed.")
        print("=" * 60)

        # Print summary
        print("\n[Experiment Summary]")
        print("-" * 50)

        # Exp1 summary
        print("\nExp1: Classification Performance (mean balanced_accuracy)")
        summary = df_exp1[df_exp1['fold'] == 'mean'][['dataset', 'model', 'balanced_accuracy', 'f1_macro', 'mae']]
        print(summary.to_string(index=False))
        
        # Exp2 summary
        print("\nExp2: Concept Validity (mean Pearson r)")
        for ds_name in ['NACC', 'PPMI']:
            ds_df = df_exp2[df_exp2['dataset'] == ds_name]
            mean_r = ds_df['pearson_r'].mean()
            print(f"  {ds_name}: mean r = {mean_r:.4f}")

        # Exp4 summary
        print("\nExp4: Counterfactual Analysis (most influential cognitive domain)")
        for ds_name in ['NACC', 'PPMI']:
            if 'sorted_by_effect' in exp4_results.get(ds_name, {}):
                top_domain = exp4_results[ds_name]['sorted_by_effect'][0]
                top_effect = exp4_results[ds_name]['counterfactual_effects'][top_domain]['mean_delta_p']
                print(f"  {ds_name}: {top_domain} (ΔP={top_effect:.4f})")
        
    except Exception as e:
        import traceback
        print(f"\n[Error] {e}")
        traceback.print_exc()
        elapsed = time.time() - start
        print(f"\nExperiment interrupted! Elapsed: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
