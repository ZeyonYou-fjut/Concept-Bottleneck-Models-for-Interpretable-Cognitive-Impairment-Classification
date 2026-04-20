"""
Experiment 5: Fairness analysis and bias correction.
Evaluates education-subgroup BA disparity and applies reweighting/calibration.

Implements three fairness correction methods on the NACC dataset,
evaluating the Balanced Accuracy gap (Fairness Gap) between
low-education (<=12 years) and high-education (>12 years) subgroups.

Methods:
1. Baseline               : No correction; evaluate CBM-Supervised by education subgroup
2. Sample Reweighting     : Assign higher weights to low-education samples (inverse-proportion weighting)
3. Education Calibration  : Add education-year adjustment at the concept layer (mimicking MoCA education correction rule)

Dataset: NACC (load_nacc_extended, 3-class classification)
CV    : Stratified 10-fold, random_state=42
Metrics: overall BA, low_edu BA, high_edu BA, Fairness Gap
Output: ./results/fairness_results.json

Corresponds to Section 4.3 in the paper.
"""

import os
import sys
import io
import json
import time
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

# Force stdout to use UTF-8 encoding (for Windows PowerShell redirection compatibility)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import balanced_accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ─── Project imports ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from run_explore_cbm import (
    COGNITIVE_DOMAINS, MOCA_FEATURES, DOMAIN_MAX_SCORES,
    CBM_Supervised, train_cbm,
    preprocess_with_concepts, compute_true_concepts,
    SEED
)
from experiment_data import load_nacc_extended, EXTENDED_FEATURES

# ─── Constants ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
EDU_THRESH   = 12      # Education year threshold for subgroup split
N_SPLITS     = 10      # 10-fold CV
EDU_IDX      = EXTENDED_FEATURES.index('EDUCATION')  # Column index of EDUCATION in feature vector

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ──────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────

def _subgroup_ba(y_true: np.ndarray, y_pred: np.ndarray,
                 edu_arr: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute overall BA + low/high education subgroup BA + fairness gap.

    Args:
        y_true  : true labels (N,)
        y_pred  : predicted labels (N,)
        edu_arr : education years array (N,), may contain NaN

    Returns:
        overall_ba, low_edu_ba, high_edu_ba, gap
    """
    overall_ba = balanced_accuracy_score(y_true, y_pred)

    # Filter NaN
    valid = ~np.isnan(edu_arr)
    low_mask  = valid & (edu_arr <= EDU_THRESH)
    high_mask = valid & (edu_arr >  EDU_THRESH)

    low_ba  = balanced_accuracy_score(y_true[low_mask],  y_pred[low_mask])  \
              if low_mask.sum() >= 10 else float('nan')
    high_ba = balanced_accuracy_score(y_true[high_mask], y_pred[high_mask]) \
              if high_mask.sum() >= 10 else float('nan')

    if not np.isnan(low_ba) and not np.isnan(high_ba):
        gap = float(high_ba - low_ba)
    else:
        gap = float('nan')

    return float(overall_ba), float(low_ba), float(high_ba), gap


def _preprocess(X_train: np.ndarray, X_test: np.ndarray):
    """KNNImputer + StandardScaler; also returns true concepts."""
    imputer = KNNImputer(n_neighbors=5)
    X_tr_imp = imputer.fit_transform(X_train)
    X_te_imp = imputer.transform(X_test)

    tc_train = compute_true_concepts(X_tr_imp)
    tc_test  = compute_true_concepts(X_te_imp)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_imp)
    X_te_s = scaler.transform(X_te_imp)

    return X_tr_s, X_te_s, tc_train, tc_test, imputer, scaler


# ──────────────────────────────────────────────────────────
# Training helper: single-pass training with sample weights
# ──────────────────────────────────────────────────────────

def _train_cbm_weighted(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        sample_weights: np.ndarray = None,
                        epochs: int = 100, batch_size: int = 32,
                        lr: float = 1e-3,
                        true_concepts_train: np.ndarray = None,
                        true_concepts_test: np.ndarray = None,
                        lambda_concept: float = 0.5,
                        lambda_polarity: float = 0.1,
                        verbose: bool = False) -> Dict[str, Any]:
    """
    Train CBM-Supervised with sample weights.
    If sample_weights is None, falls back to uniform weights (standard training).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CBM_Supervised().to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    X_tr_t = torch.FloatTensor(X_train).to(device)
    y_tr_t = torch.LongTensor(y_train).to(device)
    X_te_t = torch.FloatTensor(X_test).to(device)

    # Sample weight Tensor
    if sample_weights is None:
        w = torch.ones(len(X_train), dtype=torch.float32).to(device)
    else:
        w = torch.FloatTensor(sample_weights / sample_weights.sum() * len(sample_weights)).to(device)

    if true_concepts_train is not None:
        tc_tr_t = torch.FloatTensor(true_concepts_train).to(device)
    else:
        tc_tr_t = None

    indices = torch.arange(len(X_train), dtype=torch.long).to(device)
    dataset = TensorDataset(X_tr_t, y_tr_t, indices)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Class weights (balance class imbalance, stacked with sample weights)
    class_counts = np.bincount(y_train, minlength=3)
    total = len(y_train)
    num_cls = len(class_counts)
    class_w = torch.FloatTensor([total / (num_cls * c) if c > 0 else 1.0
                                  for c in class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w, reduction='none')

    best_loss = float('inf')
    patience_cnt = 0
    best_state = None

    model.train()
    for epoch in range(epochs):
        total_loss_ep = 0.0
        for xb, yb, idx in loader:
            optimizer.zero_grad()
            log_probs, concepts = model(xb)

            # Weighted CE loss
            ce_loss = criterion(log_probs, yb)           # (batch,)
            wb = w[idx]
            loss = (ce_loss * wb).mean()

            # Concept supervision
            if tc_tr_t is not None:
                tc_b = tc_tr_t[idx]
                concept_loss = F.mse_loss(concepts, tc_b, reduction='mean')

                # Polarity constraint
                corr_penalty = 0.0
                for d in range(concepts.shape[1]):
                    pc = concepts[:, d] - concepts[:, d].mean()
                    tc = tc_b[:, d]    - tc_b[:, d].mean()
                    corr = (pc * tc).sum() / (pc.norm() * tc.norm() + 1e-8)
                    corr_penalty += F.relu(-corr)

                concept_loss = concept_loss + lambda_polarity * corr_penalty
                loss = loss + lambda_concept * concept_loss

            loss.backward()
            optimizer.step()
            total_loss_ep += loss.item()

        scheduler.step(total_loss_ep)

        # Early stopping
        if total_loss_ep < best_loss:
            best_loss = total_loss_ep
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= 15:
                break

        if verbose and epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}: loss={total_loss_ep:.4f}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Inference
    model.eval()
    with torch.no_grad():
        log_probs_te, concepts_te = model(X_te_t)
        y_pred = torch.argmax(log_probs_te, dim=1).cpu().numpy()

    ba = balanced_accuracy_score(y_test, y_pred)
    return {'balanced_accuracy': ba, 'y_pred': y_pred, 'model': model}


# ──────────────────────────────────────────────────────────
# Method 1: Baseline (no correction)
# ──────────────────────────────────────────────────────────

def run_baseline(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Baseline: Standard CBM-Supervised, Stratified 10-fold CV.
    Collects predictions by education subgroup and computes BA.
    """
    print("\n" + "=" * 60)
    print("[Method 1] Baseline (no correction)")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_y_true = []
    all_y_pred = []
    all_edu    = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        edu_te     = X_te[:, EDU_IDX].copy()   # Education years (before standardization)

        X_tr_s, X_te_s, tc_tr, tc_te, _, _ = _preprocess(X_tr, X_te)

        result = _train_cbm_weighted(
            X_tr_s, y_tr, X_te_s, y_te,
            sample_weights=None,
            true_concepts_train=tc_tr,
            true_concepts_test=tc_te
        )
        y_pred = result['y_pred']

        # Read education years from original unstandardized data
        # _preprocess internally imputes data, but EDU_IDX values may change due to imputation
        # Here we read from original X_te's EDU_IDX column (before imputation)
        all_y_true.append(y_te)
        all_y_pred.append(y_pred)
        all_edu.append(edu_te)

        ba_fold = balanced_accuracy_score(y_te, y_pred)
        print(f"  Fold {fold+1:2d}: BA={ba_fold:.4f}")

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    all_edu    = np.concatenate(all_edu)

    overall_ba, low_ba, high_ba, gap = _subgroup_ba(all_y_true, all_y_pred, all_edu)

    print(f"\n  Overall BA  = {overall_ba:.4f}")
    print(f"  Low-edu BA  = {low_ba:.4f}")
    print(f"  High-edu BA = {high_ba:.4f}")
    print(f"  Gap         = {gap:.4f}")

    return {
        'overall_BA': overall_ba,
        'low_edu_BA': low_ba,
        'high_edu_BA': high_ba,
        'gap': gap
    }


# ──────────────────────────────────────────────────────────
# Method 2: Sample Reweighting
# ──────────────────────────────────────────────────────────

def _compute_sample_weights(edu_arr: np.ndarray) -> np.ndarray:
    """
    Inverse-proportion weighting:
      - Low-education samples (<=12 years) weight = N / (2 * N_low)
      - High-education samples (>12 years) weight = N / (2 * N_high)
    Samples with NaN education years receive uniform weight 1.0.
    """
    N = len(edu_arr)
    weights = np.ones(N, dtype=np.float32)

    valid = ~np.isnan(edu_arr)
    low_mask  = valid & (edu_arr <= EDU_THRESH)
    high_mask = valid & (edu_arr >  EDU_THRESH)

    N_low  = low_mask.sum()
    N_high = high_mask.sum()

    if N_low > 0:
        weights[low_mask]  = N / (2.0 * N_low)
    if N_high > 0:
        weights[high_mask] = N / (2.0 * N_high)

    return weights


def run_reweighting(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Sample Reweighting: Assign higher training weights to low-education samples, Stratified 10-fold CV.
    """
    print("\n" + "=" * 60)
    print("[Method 2] Sample Reweighting")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_y_true = []
    all_y_pred = []
    all_edu    = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        edu_tr     = X_tr[:, EDU_IDX].copy()
        edu_te     = X_te[:, EDU_IDX].copy()

        # Compute training sample weights (using original edu column)
        sw = _compute_sample_weights(edu_tr)

        X_tr_s, X_te_s, tc_tr, tc_te, _, _ = _preprocess(X_tr, X_te)

        result = _train_cbm_weighted(
            X_tr_s, y_tr, X_te_s, y_te,
            sample_weights=sw,
            true_concepts_train=tc_tr,
            true_concepts_test=tc_te
        )
        y_pred = result['y_pred']

        all_y_true.append(y_te)
        all_y_pred.append(y_pred)
        all_edu.append(edu_te)

        ba_fold = balanced_accuracy_score(y_te, y_pred)
        print(f"  Fold {fold+1:2d}: BA={ba_fold:.4f}")

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    all_edu    = np.concatenate(all_edu)

    overall_ba, low_ba, high_ba, gap = _subgroup_ba(all_y_true, all_y_pred, all_edu)

    print(f"\n  Overall BA  = {overall_ba:.4f}")
    print(f"  Low-edu BA  = {low_ba:.4f}")
    print(f"  High-edu BA = {high_ba:.4f}")
    print(f"  Gap         = {gap:.4f}")

    return {
        'overall_BA': overall_ba,
        'low_edu_BA': low_ba,
        'high_edu_BA': high_ba,
        'gap': gap
    }


# ──────────────────────────────────────────────────────────
# Method 3: Education-Aware Concept Calibration
# ──────────────────────────────────────────────────────────

def _apply_edu_calibration(X_imp: np.ndarray,
                            edu_raw: np.ndarray,
                            add_score: float = 1.0 / 30.0) -> np.ndarray:
    """
    Apply additive calibration to the concept layer for low-education samples (<=12 years).

    Mimicking the MoCA education correction rule: low-education individuals receive +1 point on total score.
    Here, +1/30 is uniformly distributed across 7 cognitive domain concept scores (each gets +1/30/7 ~ 0.00476),
    then clipped to [0, 1].

    Args:
        X_imp     : imputed but unscaled feature matrix (N, 24)
        edu_raw   : raw education years array (N,) (aligned with X_imp rows)
        add_score : increment added to the overall MoCA-equivalent concept (default 1/30)

    Returns:
        true_concepts_calib : calibrated true concepts (N, 7)
    """
    tc = compute_true_concepts(X_imp)  # (N, 7)
    n_domains = tc.shape[1]
    increment = add_score / n_domains  # uniformly distribute across domains

    low_mask = ~np.isnan(edu_raw) & (edu_raw <= EDU_THRESH)
    tc[low_mask] = np.clip(tc[low_mask] + increment, 0.0, 1.0)

    return tc.astype(np.float32)


def run_education_calibration(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Education-Aware Concept Calibration:
    When computing true concept supervision targets, apply +1/30 correction to concept values
    of low-education samples, so CBM concept predictors learn fairer concept targets.
    Stratified 10-fold CV.
    """
    print("\n" + "=" * 60)
    print("[Method 3] Education-Aware Concept Calibration")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_y_true = []
    all_y_pred = []
    all_edu    = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        edu_tr = X_tr[:, EDU_IDX].copy()
        edu_te = X_te[:, EDU_IDX].copy()

        # Step 1: Impute
        imputer = KNNImputer(n_neighbors=5)
        X_tr_imp = imputer.fit_transform(X_tr)
        X_te_imp = imputer.transform(X_te)

        # Step 2: Compute calibrated concept supervision targets (training set)
        tc_tr_calib = _apply_edu_calibration(X_tr_imp, edu_tr)
        # Test set not calibrated (keep original standard for evaluation)
        tc_te_std   = compute_true_concepts(X_te_imp)

        # Step 3: Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_imp)
        X_te_s = scaler.transform(X_te_imp)

        # Step 4: Train
        result = _train_cbm_weighted(
            X_tr_s, y_tr, X_te_s, y_te,
            sample_weights=None,
            true_concepts_train=tc_tr_calib,
            true_concepts_test=tc_te_std
        )
        y_pred = result['y_pred']

        all_y_true.append(y_te)
        all_y_pred.append(y_pred)
        all_edu.append(edu_te)

        ba_fold = balanced_accuracy_score(y_te, y_pred)
        print(f"  Fold {fold+1:2d}: BA={ba_fold:.4f}")

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    all_edu    = np.concatenate(all_edu)

    overall_ba, low_ba, high_ba, gap = _subgroup_ba(all_y_true, all_y_pred, all_edu)

    print(f"\n  Overall BA  = {overall_ba:.4f}")
    print(f"  Low-edu BA  = {low_ba:.4f}")
    print(f"  High-edu BA = {high_ba:.4f}")
    print(f"  Gap         = {gap:.4f}")

    return {
        'overall_BA': overall_ba,
        'low_edu_BA': low_ba,
        'high_edu_BA': high_ba,
        'gap': gap
    }


# ──────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Fairness Correction Experiment - Mitigating BA Degradation in Low-Education Subgroup")
    print("=" * 60)
    print(f"Start time    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random seed   : {SEED}")
    print(f"Edu threshold : <= {EDU_THRESH} years (low-education)")
    print(f"CV folds      : {N_SPLITS}-fold Stratified")
    print(f"EDUCATION col index: {EDU_IDX}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    start = time.time()

    try:
        # ── Load data ──────────────────────────────────────
        print("\n[Data Loading]")
        print("-" * 50)
        X, y = load_nacc_extended(max_per_class=500)
        print(f"  Data shape: X={X.shape}, y={y.shape}")
        print(f"  Class distribution: {dict(zip(['Normal','MCI','Dementia'], np.bincount(y)))}")

        edu_col = X[:, EDU_IDX]
        valid_edu = edu_col[~np.isnan(edu_col)]
        low_cnt  = (valid_edu <= EDU_THRESH).sum()
        high_cnt = (valid_edu >  EDU_THRESH).sum()
        print(f"  Education distribution: low-edu(<=12y)={low_cnt}, high-edu(>12y)={high_cnt}")
        print(f"  Education range: {np.nanmin(edu_col):.0f} - {np.nanmax(edu_col):.0f}, "
              f"mean={np.nanmean(edu_col):.1f}")

        # ── Method 1: Baseline ───────────────────────────────
        res_baseline = run_baseline(X, y)

        # ── Method 2: Sample Reweighting ────────────────────
        res_reweight = run_reweighting(X, y)

        # ── Method 3: Education Calibration ─────────────────
        res_calib    = run_education_calibration(X, y)

        # ── Aggregate results ───────────────────────────────
        results = {
            'baseline': res_baseline,
            'reweighting': res_reweight,
            'education_calibration': res_calib
        }

        # Save JSON
        output_path = os.path.join(RESULTS_DIR, 'fairness_results.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start

        # ── Print summary ───────────────────────────────────
        print("\n" + "=" * 60)
        print("Experiment Completion Summary")
        print("=" * 60)
        header = f"{'Method':<28} {'Overall BA':>10} {'Low-edu BA':>10} {'High-edu BA':>11} {'Gap':>8}"
        print(header)
        print("-" * 70)
        for method_name, res in [
            ("Baseline",              res_baseline),
            ("Sample Reweighting",    res_reweight),
            ("Education Calibration", res_calib)
        ]:
            print(f"{method_name:<28} "
                  f"{res['overall_BA']:>10.4f} "
                  f"{res['low_edu_BA']:>10.4f} "
                  f"{res['high_edu_BA']:>11.4f} "
                  f"{res['gap']:>8.4f}")

        print("\nImprovement (relative to Baseline Gap):")
        base_gap = res_baseline['gap']
        for method_name, res in [
            ("Sample Reweighting",    res_reweight),
            ("Education Calibration", res_calib)
        ]:
            delta = base_gap - res['gap']
            pct   = (delta / abs(base_gap) * 100) if base_gap != 0 else 0.0
            arrow = "↓" if delta > 0 else "↑"
            print(f"  {method_name:<28}: Gap {arrow} {abs(delta):.4f} ({pct:+.1f}%)")

        print(f"\nTotal time   : {elapsed/60:.1f} minutes")
        print(f"Results saved: {output_path}")
        print("=" * 60)

    except Exception as e:
        import traceback
        print(f"\n[Error] {e}")
        traceback.print_exc()
        elapsed = time.time() - start
        print(f"Experiment interrupted, elapsed: {elapsed/60:.1f} minutes")
        raise


if __name__ == '__main__':
    main()
