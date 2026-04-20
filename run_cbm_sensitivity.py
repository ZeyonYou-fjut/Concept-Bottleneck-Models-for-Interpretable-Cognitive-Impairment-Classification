"""
Experiment 10: Hyperparameter sensitivity analysis.
Evaluates model robustness to learning rate, hidden dimension, and regularization changes.
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
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer
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
from run_explore_cbm import (
    COGNITIVE_DOMAINS, MOCA_FEATURES, DOMAIN_MAX_SCORES,
    compute_true_concepts, CBM_Supervised, train_cbm
)
from experiment_data import (
    load_nacc_extended, load_ppmi_extended, EXTENDED_FEATURES
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


def preprocess_with_concepts_custom(
    X_train: np.ndarray, 
    X_test: np.ndarray,
    imputer,
    scaler
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Custom preprocessing function supporting different imputers and scalers.

    Key: compute_true_concepts must be computed after imputation, before scaling.

    Args:
        X_train, X_test: raw feature matrices
        imputer: missing value imputer instance
        scaler: feature scaler instance
        
    Returns:
        X_train_scaled, X_test_scaled: preprocessed features
        true_concepts_train, true_concepts_test: normalized cognitive domain scores
    """
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # Compute true concepts after imputation but before scaling
    true_concepts_train = compute_true_concepts(X_train_imp)
    true_concepts_test = compute_true_concepts(X_test_imp)
    
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    return X_train_scaled, X_test_scaled, true_concepts_train, true_concepts_test


def compute_concept_validity(pred_concepts: np.ndarray, true_concepts: np.ndarray) -> Dict[str, float]:
    """
    Compute concept validity: Pearson correlation between predicted and true concepts.

    Args:
        pred_concepts: predicted concepts, shape=(N, 7)
        true_concepts: true concepts, shape=(N, 7)
        
    Returns:
        dict: per-domain Pearson r and mean
    """
    domain_names = list(COGNITIVE_DOMAINS.keys())
    validity_by_domain = {}
    
    for i, domain in enumerate(domain_names):
        r, p = pearsonr(pred_concepts[:, i], true_concepts[:, i])
        validity_by_domain[domain] = float(r)
    
    mean_r = np.mean(list(validity_by_domain.values()))
    
    return {
        'mean_r': float(mean_r),
        'by_domain': validity_by_domain
    }


def run_cbm_cv_with_config(
    X: np.ndarray,
    y: np.ndarray,
    imputer,
    scaler,
    lambda_concept: float = 0.5,
    lr: float = 1e-3,
    batch_size: int = 32,
    config_name: str = "config"
) -> Dict[str, Any]:
    """
    Run CBM-Supervised 5-fold CV with specified configuration.

    Args:
        X, y: data
        imputer: missing value imputer
        scaler: feature scaler
        lambda_concept: concept supervision weight
        lr: learning rate
        batch_size: batch size
        config_name: configuration name (for printing)
        
    Returns:
        dict: CV results
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    fold_bas = []
    fold_f1s = []
    all_pred_concepts = []
    all_true_concepts = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Preprocessing
        X_train_s, X_test_s, tc_train, tc_test = preprocess_with_concepts_custom(
            X_train, X_test, imputer, scaler
        )
        
        # Train CBM-Supervised
        model = CBM_Supervised()
        res = train_cbm(
            model, X_train_s, y_train, X_test_s, y_test,
            concept_supervision=True,
            true_concepts_train=tc_train,
            true_concepts_test=tc_test,
            lambda_concept=lambda_concept,
            lr=lr,
            batch_size=batch_size,
            lambda_polarity=0.1
        )
        
        fold_bas.append(res['balanced_accuracy'])
        fold_f1s.append(res['f1_macro'])
        all_pred_concepts.append(res['concepts'])
        all_true_concepts.append(tc_test)
    
    # Aggregate results
    all_pred_concepts = np.vstack(all_pred_concepts)
    all_true_concepts = np.vstack(all_true_concepts)
    
    concept_validity = compute_concept_validity(all_pred_concepts, all_true_concepts)
    
    return {
        'mean_ba': float(np.mean(fold_bas)),
        'std_ba': float(np.std(fold_bas)),
        'mean_f1': float(np.mean(fold_f1s)),
        'std_f1': float(np.std(fold_f1s)),
        'concept_validity_mean_r': concept_validity['mean_r'],
        'concept_validity_by_domain': concept_validity['by_domain'],
        'fold_bas': [float(x) for x in fold_bas],
        'fold_f1s': [float(x) for x in fold_f1s]
    }


def run_exp11a_preprocessing_sensitivity(X_nacc: np.ndarray, y_nacc: np.ndarray) -> List[Dict]:
    """
    Exp-11a: Preprocessing method sensitivity (NACC).
    
    6 preprocessing configurations:
    P1: KNNImputer(5) + StandardScaler (baseline)
    P2: KNNImputer(3) + StandardScaler
    P3: KNNImputer(10) + StandardScaler
    P4: SimpleImputer(mean) + StandardScaler
    P5: KNNImputer(5) + RobustScaler
    P6: KNNImputer(5) + MinMaxScaler
    """
    print("\n" + "=" * 60)
    print("Exp-11a: Preprocessing Method Sensitivity (NACC)")
    print("=" * 60)
    
    configs = [
        ("P1_KNN5_Standard", KNNImputer(n_neighbors=5), StandardScaler()),
        ("P2_KNN3_Standard", KNNImputer(n_neighbors=3), StandardScaler()),
        ("P3_KNN10_Standard", KNNImputer(n_neighbors=10), StandardScaler()),
        ("P4_SimpleMean_Standard", SimpleImputer(strategy='mean'), StandardScaler()),
        ("P5_KNN5_Robust", KNNImputer(n_neighbors=5), RobustScaler()),
        ("P6_KNN5_MinMax", KNNImputer(n_neighbors=5), MinMaxScaler()),
    ]
    
    results = []
    
    for config_name, imputer, scaler in configs:
        print(f"\n[{config_name}] Running...")
        
        result = run_cbm_cv_with_config(
            X_nacc, y_nacc, imputer, scaler,
            lambda_concept=0.5, lr=1e-3, batch_size=32,
            config_name=config_name
        )
        
        result_entry = {
            'config': config_name,
            'imputer': str(imputer),
            'scaler': str(scaler),
            'dataset': 'NACC',
            'mean_ba': result['mean_ba'],
            'std_ba': result['std_ba'],
            'mean_f1': result['mean_f1'],
            'std_f1': result['std_f1'],
            'concept_validity_mean_r': result['concept_validity_mean_r'],
            'concept_validity_by_domain': result['concept_validity_by_domain']
        }
        results.append(result_entry)
        
        print(f"  BA: {result['mean_ba']:.4f} ± {result['std_ba']:.4f}")
        print(f"  F1: {result['mean_f1']:.4f} ± {result['std_f1']:.4f}")
        print(f"  Concept r: {result['concept_validity_mean_r']:.4f}")
    
    return results


def run_exp11b_hyperparams_sensitivity(X_nacc: np.ndarray, y_nacc: np.ndarray) -> List[Dict]:
    """
    Exp-11b: Model hyperparameter sensitivity (NACC).
    
    Fixed P1 preprocessing, varying hyperparams:
    H1: lambda=0.5, lr=1e-3, bs=32 (baseline)
    H2: lambda=0.1, lr=1e-3, bs=32
    H3: lambda=1.0, lr=1e-3, bs=32
    H4: lambda=0.5, lr=5e-4, bs=32
    H5: lambda=0.5, lr=1e-3, bs=64
    H6: lambda=0.5, lr=1e-3, bs=16
    """
    print("\n" + "=" * 60)
    print("Exp-11b: Model Hyperparameter Sensitivity (NACC)")
    print("=" * 60)
    
    configs = [
        ("H1_lambda0.5_lr1e-3_bs32", 0.5, 1e-3, 32),
        ("H2_lambda0.1_lr1e-3_bs32", 0.1, 1e-3, 32),
        ("H3_lambda1.0_lr1e-3_bs32", 1.0, 1e-3, 32),
        ("H4_lambda0.5_lr5e-4_bs32", 0.5, 5e-4, 32),
        ("H5_lambda0.5_lr1e-3_bs64", 0.5, 1e-3, 64),
        ("H6_lambda0.5_lr1e-3_bs16", 0.5, 1e-3, 16),
    ]
    
    results = []
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()
    
    for config_name, lambda_c, lr, bs in configs:
        print(f"\n[{config_name}] Running...")
        
        result = run_cbm_cv_with_config(
            X_nacc, y_nacc, imputer, scaler,
            lambda_concept=lambda_c, lr=lr, batch_size=bs,
            config_name=config_name
        )
        
        result_entry = {
            'config': config_name,
            'lambda_concept': lambda_c,
            'lr': lr,
            'batch_size': bs,
            'dataset': 'NACC',
            'mean_ba': result['mean_ba'],
            'std_ba': result['std_ba'],
            'mean_f1': result['mean_f1'],
            'std_f1': result['std_f1'],
            'concept_validity_mean_r': result['concept_validity_mean_r'],
            'concept_validity_by_domain': result['concept_validity_by_domain']
        }
        results.append(result_entry)
        
        print(f"  BA: {result['mean_ba']:.4f} ± {result['std_ba']:.4f}")
        print(f"  F1: {result['mean_f1']:.4f} ± {result['std_f1']:.4f}")
        print(f"  Concept r: {result['concept_validity_mean_r']:.4f}")
    
    return results


def run_exp11c_ppmi_validation(X_ppmi: np.ndarray, y_ppmi: np.ndarray) -> List[Dict]:
    """
    Exp-11c: PPMI cross-validation.
    
    Run P1/P4/P5 preprocessing + H1 hyperparams on PPMI data.
    """
    print("\n" + "=" * 60)
    print("Exp-11c: PPMI Cross-Validation")
    print("=" * 60)
    
    configs = [
        ("P1_KNN5_Standard", KNNImputer(n_neighbors=5), StandardScaler()),
        ("P4_SimpleMean_Standard", SimpleImputer(strategy='mean'), StandardScaler()),
        ("P5_KNN5_Robust", KNNImputer(n_neighbors=5), RobustScaler()),
    ]
    
    results = []
    
    for config_name, imputer, scaler in configs:
        print(f"\n[{config_name}] Running...")
        
        result = run_cbm_cv_with_config(
            X_ppmi, y_ppmi, imputer, scaler,
            lambda_concept=0.5, lr=1e-3, batch_size=32,
            config_name=config_name
        )
        
        result_entry = {
            'config': config_name,
            'imputer': str(imputer),
            'scaler': str(scaler),
            'dataset': 'PPMI',
            'mean_ba': result['mean_ba'],
            'std_ba': result['std_ba'],
            'mean_f1': result['mean_f1'],
            'std_f1': result['std_f1'],
            'concept_validity_mean_r': result['concept_validity_mean_r'],
            'concept_validity_by_domain': result['concept_validity_by_domain']
        }
        results.append(result_entry)
        
        print(f"  BA: {result['mean_ba']:.4f} ± {result['std_ba']:.4f}")
        print(f"  F1: {result['mean_f1']:.4f} ± {result['std_f1']:.4f}")
        print(f"  Concept r: {result['concept_validity_mean_r']:.4f}")
    
    return results


def compute_cv_metric(results: List[Dict], metric_key: str = 'mean_ba') -> float:
    """
    Compute coefficient of variation (CV = std / mean).

    Args:
        results: results list
        metric_key: metric key name
        
    Returns:
        float: coefficient of variation
    """
    values = [r[metric_key] for r in results]
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    if mean_val == 0:
        return float('inf')
    
    return float(std_val / mean_val)


def main():
    print("=" * 60)
    print("CBM Sensitivity Analysis Experiment (Exp-10)")
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
        
        # ========== Exp-11a: Preprocessing Method Sensitivity ==========
        exp11a_results = run_exp11a_preprocessing_sensitivity(X_nacc, y_nacc)
        
        # ========== Exp-11b: Hyperparameter Sensitivity ==========
        exp11b_results = run_exp11b_hyperparams_sensitivity(X_nacc, y_nacc)
        
        # ========== Exp-11c: PPMI Cross-Validation ==========
        exp11c_results = run_exp11c_ppmi_validation(X_ppmi, y_ppmi)
        
        # ========== Compute Summary ==========
        preprocessing_ba_cv = compute_cv_metric(exp11a_results, 'mean_ba')
        hyperparams_ba_cv = compute_cv_metric(exp11b_results, 'mean_ba')
        ppmi_preprocessing_ba_cv = compute_cv_metric(exp11c_results, 'mean_ba')
        
        summary = {
            'preprocessing_ba_cv': preprocessing_ba_cv,
            'hyperparams_ba_cv': hyperparams_ba_cv,
            'ppmi_preprocessing_ba_cv': ppmi_preprocessing_ba_cv
        }
        
        # ========== Save Results ==========
        output = {
            'exp11a_preprocessing': exp11a_results,
            'exp11b_hyperparams': exp11b_results,
            'exp11c_ppmi': exp11c_results,
            'summary': summary
        }
        
        output_path = os.path.join(RESULTS_DIR, 'cbm_exp11_sensitivity.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        # ========== Done ==========
        elapsed = time.time() - start
        print("\n" + "=" * 60)
        print(f"Total time: {elapsed/60:.1f} min")
        print(f"Results saved to: {output_path}")
        print("=" * 60)
        
        # ========== Print Summary ==========
        print("\n[Experiment Summary]")
        print("-" * 50)
        
        print("\nExp-11a: Preprocessing Method Sensitivity (NACC)")
        print(f"  BA CV: {preprocessing_ba_cv:.4f} ({'<0.03 highly stable' if preprocessing_ba_cv < 0.03 else 'needs attention'})")
        for r in exp11a_results:
            print(f"    {r['config']}: BA={r['mean_ba']:.4f} ± {r['std_ba']:.4f}, r={r['concept_validity_mean_r']:.4f}")
        
        print("\nExp-11b: Hyperparameter Sensitivity (NACC)")
        print(f"  BA CV: {hyperparams_ba_cv:.4f} ({'<0.03 highly stable' if hyperparams_ba_cv < 0.03 else 'needs attention'})")
        for r in exp11b_results:
            print(f"    {r['config']}: BA={r['mean_ba']:.4f} ± {r['std_ba']:.4f}, r={r['concept_validity_mean_r']:.4f}")
        
        print("\nExp-11c: PPMI Cross-Validation")
        print(f"  BA CV: {ppmi_preprocessing_ba_cv:.4f} ({'<0.03 highly stable' if ppmi_preprocessing_ba_cv < 0.03 else 'needs attention'})")
        for r in exp11c_results:
            print(f"    {r['config']}: BA={r['mean_ba']:.4f} ± {r['std_ba']:.4f}, r={r['concept_validity_mean_r']:.4f}")
        
        print("\n" + "=" * 60)
        print("CBM Sensitivity Analysis Experiment Complete!")
        print("=" * 60)
        
        # Verify results file was generated
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"\n[Confirmed] Results file generated: {output_path}")
            print(f"  File size: {file_size} bytes")
        else:
            print(f"\n[WARNING] Results file not found: {output_path}")
        
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        elapsed = time.time() - start
        print(f"\nExperiment interrupted! Time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
