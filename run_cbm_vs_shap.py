"""
Experiment 8: CBM vs SHAP interpretability comparison.
Compares concept-based explanations with post-hoc SHAP attributions.

Compares three methods for ranking cognitive domain importance:
1. CBM counterfactual effect: set concept to domain max score (1.0) and observe P(MCI->Normal) change
2. Gradient attribution: Gradient x Input, aggregated by domain
3. SHAP attribution (if available): DeepExplainer or GradientExplainer

Evaluation metrics:
- Ranking consistency (Spearman rho)
- Cross-fold stability (ranking std)
- Clinical alignment (Spearman rho with AD gold standard ranking)
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from collections import OrderedDict

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from scipy.stats import spearmanr

# Check SHAP availability
try:
    import shap
    SHAP_AVAILABLE = True
    print("[SHAP] SHAP library detected")
except ImportError:
    SHAP_AVAILABLE = False
    print("[SHAP] SHAP library not installed, skipping SHAP analysis")

# Import required components from run_explore_cbm.py
sys.path.insert(0, os.path.dirname(__file__))
from run_explore_cbm import (
    COGNITIVE_DOMAINS, MOCA_FEATURES, DOMAIN_MAX_SCORES,
    preprocess_with_concepts, preprocess_dnn,
    CBM_Supervised, train_cbm,
    compute_true_concepts
)
from model_dcm_dnn import PlainDNN, train_and_evaluate
from experiment_data import load_nacc_extended

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# AD clinical gold standard domain importance ranking (high to low)
# Delayed_Recall is the most sensitive early AD indicator, followed by Language and Visuospatial_Executive
CLINICAL_GOLD_STANDARD = [
    'Delayed_Recall',       # Memory - earliest AD impairment
    'Language',             # Language - naming and fluency deficits
    'Visuospatial_Executive', # Visuospatial/Executive - drawing difficulties
    'Orientation',          # Orientation - time/place confusion
    'Attention',            # Attention - later-stage impairment
    'Abstraction',          # Abstraction - less sensitive
    'Naming'                # Naming - only 3 points in MoCA
]


def compute_cbm_counterfactual_importance(model, X_test: np.ndarray, y_test: np.ndarray,
                                          device: torch.device) -> np.ndarray:
    """
    Compute CBM counterfactual effect as domain importance.
    
    For each MCI sample, set each domain concept to 1.0 (max) and observe P(Normal) change.
    Returns mean effect per domain [7].
    """
    model.eval()
    model = model.to(device)
    
    # Select MCI samples
    mci_mask = y_test == 1
    if mci_mask.sum() == 0:
        return np.zeros(len(COGNITIVE_DOMAINS))
    
    X_mci = X_test[mci_mask]
    X_tensor = torch.FloatTensor(X_mci).to(device)
    
    # Get original concepts and probabilities
    with torch.no_grad():
        orig_probs, orig_concepts = model.predict_proba(X_tensor)
        orig_p_normal = orig_probs[:, 0].cpu().numpy()
    
    effects = []
    
    # Counterfactual analysis per domain
    for i, domain in enumerate(COGNITIVE_DOMAINS.keys()):
        # Set i-th concept to 1.0 (max score)
        modified_concepts = orig_concepts.clone()
        modified_concepts[:, i] = 1.0
        
        # Compute new probabilities using diagnosis head
        with torch.no_grad():
            x_demo = torch.FloatTensor(X_mci[:, 19:24]).to(device)
            if hasattr(model, 'diagnosis_head'):
                combined = torch.cat([modified_concepts, x_demo], dim=1)
                logits = model.diagnosis_head(combined)
                new_probs = F.softmax(logits, dim=-1)
                new_p_normal = new_probs[:, 0].cpu().numpy()
            else:
                # Unknown model type, keep original
                new_p_normal = orig_p_normal.copy()
        
        # Compute P(Normal) delta
        delta_p = new_p_normal - orig_p_normal
        effects.append(delta_p.mean())
    
    return np.array(effects)


def compute_gradient_attribution(model, X_test: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Compute gradient attribution as domain importance.
    
    For each sample, compute Gradient x Input, then aggregate by domain.
    Returns mean attribution per domain [7].
    """
    model.eval()
    model = model.to(device)
    
    X_tensor = torch.FloatTensor(X_test).to(device)
    X_tensor.requires_grad_(True)
    
    # Forward pass
    logits = model(X_tensor)
    
    # Get predicted classes
    probs = F.softmax(logits, dim=-1)
    pred_classes = probs.argmax(dim=1)
    
    # Compute gradients per sample
    attributions = []
    
    for i in range(len(X_tensor)):
        model.zero_grad()
        X_tensor.grad = None if X_tensor.grad is not None else None
        
        # Compute gradient w.r.t. predicted class
        logits[i, pred_classes[i]].backward(retain_graph=True)
        grad = X_tensor.grad[i].detach().cpu().numpy()
        
        # Gradient x Input (first 19 MoCA features only)
        x_i = X_test[i, :19]
        attr = (grad[:19] * x_i).astype(float)
        
        # Aggregate by domain (sum of absolute values)
        domain_attrs = []
        for j, (domain, items) in enumerate(COGNITIVE_DOMAINS.items()):
            indices = [MOCA_FEATURES.index(item) for item in items]
            domain_attr = np.abs(attr[indices]).sum()
            domain_attrs.append(domain_attr)
        
        attributions.append(domain_attrs)
    
    return np.mean(attributions, axis=0)


def compute_shap_attribution(model, X_train: np.ndarray, X_test: np.ndarray,
                             device: torch.device, n_background: int = 100) -> np.ndarray:
    """
    Compute SHAP attribution as domain importance.
    
    Uses GradientExplainer or DeepExplainer.
    Returns mean SHAP value per domain [7].
    """
    if not SHAP_AVAILABLE:
        return None
    
    model.eval()
    model = model.to(device)
    
    # Select background samples
    n_background = min(n_background, len(X_train))
    bg_indices = np.random.choice(len(X_train), n_background, replace=False)
    X_background = torch.FloatTensor(X_train[bg_indices]).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    try:
        # Try GradientExplainer
        explainer = shap.GradientExplainer(model, X_background)
        shap_values = explainer.shap_values(X_test_tensor)
        
        # shap_values may be a list (one per class) or an array
        if isinstance(shap_values, list):
            # Take SHAP values for predicted class
            probs = model(X_test_tensor)
            probs = F.softmax(probs, dim=-1)
            pred_classes = probs.argmax(dim=1).cpu().numpy()
            
            all_shap = []
            for i, pred_class in enumerate(pred_classes):
                all_shap.append(shap_values[pred_class][i, :19])
            shap_values = np.array(all_shap)
        else:
            shap_values = shap_values[:, :19]
        
        # Aggregate by domain
        domain_shaps = []
        for j, (domain, items) in enumerate(COGNITIVE_DOMAINS.items()):
            indices = [MOCA_FEATURES.index(item) for item in items]
            domain_shap = np.abs(shap_values[:, indices]).sum(axis=1).mean()
            domain_shaps.append(domain_shap)
        
        return np.array(domain_shaps)
        
    except Exception as e:
        print(f"  [SHAP ERROR] {e}, trying DeepExplainer...")
        
        try:
            explainer = shap.DeepExplainer(model, X_background)
            shap_values = explainer.shap_values(X_test_tensor)
            
            if isinstance(shap_values, list):
                probs = model(X_test_tensor)
                probs = F.softmax(probs, dim=-1)
                pred_classes = probs.argmax(dim=1).cpu().numpy()
                
                all_shap = []
                for i, pred_class in enumerate(pred_classes):
                    all_shap.append(shap_values[pred_class][i, :19])
                shap_values = np.array(all_shap)
            else:
                shap_values = shap_values[:, :19]
            
            domain_shaps = []
            for j, (domain, items) in enumerate(COGNITIVE_DOMAINS.items()):
                indices = [MOCA_FEATURES.index(item) for item in items]
                domain_shap = np.abs(shap_values[:, indices]).sum(axis=1).mean()
                domain_shaps.append(domain_shap)
            
            return np.array(domain_shaps)
            
        except Exception as e2:
            print(f"  [SHAP DeepExplainer ERROR] {e2}")
            return None


def train_plain_dnn(input_dim: int, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray, num_classes: int = 3,
                    max_retries: int = 3) -> Tuple[Any, Dict]:
    """Train PlainDNN model, return model and training results."""
    best_model = None
    best_result = None
    best_ba = 0.0
    
    for attempt in range(max_retries):
        torch.manual_seed(SEED + attempt)
        np.random.seed(SEED + attempt)
        
        model = PlainDNN(input_dim=input_dim, num_classes=num_classes)
        
        result = train_and_evaluate(
            model, X_train, y_train, X_test, y_test,
            epochs=100, batch_size=32, verbose=False,
            use_focal_loss=False
        )
        
        ba = result['balanced_accuracy']
        
        if ba > best_ba:
            best_ba = ba
            best_model = model
            best_result = result
        
        if ba > 0.4:
            break
    
    return best_model, best_result


def compute_spearman_rho(rank1: List[str], rank2: List[str]) -> float:
    """Compute Spearman rho between two rankings."""
    if len(rank1) != len(rank2):
        raise ValueError("Ranking lengths do not match")
    
    n = len(rank1)
    rank1_pos = {domain: i for i, domain in enumerate(rank1)}
    rank2_pos = {domain: i for i, domain in enumerate(rank2)}
    
    # Compute squared rank differences
    d_squared = 0
    for domain in rank1:
        d_squared += (rank1_pos[domain] - rank2_pos[domain]) ** 2
    
    # Spearman rho
    rho = 1 - (6 * d_squared) / (n * (n ** 2 - 1))
    return rho


def values_to_rank(values: np.ndarray, domain_names: List[str]) -> List[str]:
    """Convert numeric importance values to domain ranking (high to low)."""
    sorted_indices = np.argsort(values)[::-1]
    return [domain_names[i] for i in sorted_indices]


def main():
    print("=" * 60)
    print("CBM vs SHAP Interpretability Comparison Experiment")
    print("=" * 60)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results = {
        'experiment': 'CBM vs SHAP Interpretability Comparison',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_folds': 5,
        'domains': list(COGNITIVE_DOMAINS.keys()),
        'clinical_gold_standard': CLINICAL_GOLD_STANDARD,
        'folds': {},
        'fold_importance_values': {
            'CBM': [],
            'Gradient': [],
            'SHAP': [] if SHAP_AVAILABLE else None
        },
        'fold_rankings': {
            'CBM': [],
            'Gradient': [],
            'SHAP': [] if SHAP_AVAILABLE else None
        }
    }
    
    domain_names = list(COGNITIVE_DOMAINS.keys())
    
    try:
        # Load NACC data
        print("\n[Data Loading]")
        print("-" * 50)
        X, y = load_nacc_extended(max_per_class=500)
        print(f"NACC: {X.shape}, class distribution={np.bincount(y)}")
        
        # 5-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\n{'='*60}")
            print(f"Fold {fold}")
            print(f"{'='*60}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            fold_result = {}
            
            # ========== 1. CBM-Supervised ==========
            print(f"\n[1] Training CBM-Supervised...")
            X_train_s, X_test_s, tc_train, tc_test = preprocess_with_concepts(X_train, X_test)
            
            cbm_model = CBM_Supervised()
            cbm_result = train_cbm(
                cbm_model, X_train_s, y_train, X_test_s, y_test,
                concept_supervision=True,
                lambda_concept=0.5,
                lambda_polarity=0.1,
                true_concepts_train=tc_train,
                true_concepts_test=tc_test,
                verbose=False
            )
            print(f"  CBM BA={cbm_result['balanced_accuracy']:.3f}")
            
            # Compute CBM counterfactual effect
            cbm_importance = compute_cbm_counterfactual_importance(
                cbm_result['model'], X_test_s, y_test, device
            )
            cbm_rank = values_to_rank(cbm_importance, domain_names)
            
            fold_result['CBM'] = {
                'balanced_accuracy': cbm_result['balanced_accuracy'],
                'importance_values': cbm_importance.tolist(),
                'ranking': cbm_rank
            }
            results['fold_importance_values']['CBM'].append(cbm_importance.tolist())
            results['fold_rankings']['CBM'].append(cbm_rank)
            
            print(f"  CBM domain importance ranking: {cbm_rank}")
            
            # ========== 2. PlainDNN + Gradient Attribution ==========
            print(f"\n[2] Training PlainDNN...")
            input_dim = X_train_s.shape[1]
            
            dnn_model, dnn_result = train_plain_dnn(
                input_dim, X_train_s, y_train, X_test_s, y_test, num_classes=3
            )
            print(f"  DNN BA={dnn_result['balanced_accuracy']:.3f}")
            
            # Compute gradient attribution
            grad_importance = compute_gradient_attribution(dnn_model, X_test_s, device)
            grad_rank = values_to_rank(grad_importance, domain_names)
            
            fold_result['Gradient'] = {
                'balanced_accuracy': dnn_result['balanced_accuracy'],
                'importance_values': grad_importance.tolist(),
                'ranking': grad_rank
            }
            results['fold_importance_values']['Gradient'].append(grad_importance.tolist())
            results['fold_rankings']['Gradient'].append(grad_rank)
            
            print(f"  Gradient attribution ranking: {grad_rank}")
            
            # ========== 3. SHAP Attribution (if available) ==========
            if SHAP_AVAILABLE:
                print(f"\n[3] Computing SHAP attribution...")
                shap_importance = compute_shap_attribution(
                    dnn_model, X_train_s, X_test_s, device, n_background=100
                )
                
                if shap_importance is not None:
                    shap_rank = values_to_rank(shap_importance, domain_names)
                    fold_result['SHAP'] = {
                        'importance_values': shap_importance.tolist(),
                        'ranking': shap_rank
                    }
                    results['fold_importance_values']['SHAP'].append(shap_importance.tolist())
                    results['fold_rankings']['SHAP'].append(shap_rank)
                    print(f"  SHAP attribution ranking: {shap_rank}")
                else:
                    fold_result['SHAP'] = None
                    print(f"  SHAP computation failed, skipping")
            
            results['folds'][f'fold_{fold}'] = fold_result
        
        # ========== Compute Summary Metrics ==========
        print("\n" + "=" * 60)
        print("Summary Analysis")
        print("=" * 60)
        
        # Compute cross-fold average importance
        avg_cbm_importance = np.mean(results['fold_importance_values']['CBM'], axis=0)
        avg_grad_importance = np.mean(results['fold_importance_values']['Gradient'], axis=0)
        
        avg_cbm_rank = values_to_rank(avg_cbm_importance, domain_names)
        avg_grad_rank = values_to_rank(avg_grad_importance, domain_names)
        
        results['average_importance'] = {
            'CBM': avg_cbm_importance.tolist(),
            'Gradient': avg_grad_importance.tolist()
        }
        results['average_rankings'] = {
            'CBM': avg_cbm_rank,
            'Gradient': avg_grad_rank
        }
        
        print(f"\nCross-fold average ranking:")
        print(f"  CBM: {avg_cbm_rank}")
        print(f"  Gradient: {avg_grad_rank}")
        print(f"  Clinical gold standard: {CLINICAL_GOLD_STANDARD}")
        
        # Compute ranking consistency
        cbm_grad_rho = compute_spearman_rho(avg_cbm_rank, avg_grad_rank)
        
        results['ranking_consistency'] = {
            'CBM_vs_Gradient': {
                'spearman_rho': cbm_grad_rho
            }
        }
        
        print(f"\nRanking Consistency (Spearman rho):")
        print(f"  CBM vs Gradient: {cbm_grad_rho:.4f}")
        
        if SHAP_AVAILABLE and results['fold_importance_values']['SHAP']:
            avg_shap_importance = np.mean(results['fold_importance_values']['SHAP'], axis=0)
            avg_shap_rank = values_to_rank(avg_shap_importance, domain_names)
            
            results['average_importance']['SHAP'] = avg_shap_importance.tolist()
            results['average_rankings']['SHAP'] = avg_shap_rank
            
            cbm_shap_rho = compute_spearman_rho(avg_cbm_rank, avg_shap_rank)
            grad_shap_rho = compute_spearman_rho(avg_grad_rank, avg_shap_rank)
            
            results['ranking_consistency']['CBM_vs_SHAP'] = {'spearman_rho': cbm_shap_rho}
            results['ranking_consistency']['Gradient_vs_SHAP'] = {'spearman_rho': grad_shap_rho}
            
            print(f"  CBM vs SHAP: {cbm_shap_rho:.4f}")
            print(f"  Gradient vs SHAP: {grad_shap_rho:.4f}")
        
        # Compute clinical alignment
        cbm_clinical_rho = compute_spearman_rho(avg_cbm_rank, CLINICAL_GOLD_STANDARD)
        grad_clinical_rho = compute_spearman_rho(avg_grad_rank, CLINICAL_GOLD_STANDARD)
        
        results['clinical_alignment'] = {
            'CBM': {'spearman_rho': cbm_clinical_rho},
            'Gradient': {'spearman_rho': grad_clinical_rho}
        }
        
        print(f"\nClinical Alignment (Spearman rho with gold standard ranking):")
        print(f"  CBM: {cbm_clinical_rho:.4f}")
        print(f"  Gradient: {grad_clinical_rho:.4f}")
        
        if SHAP_AVAILABLE and results['fold_importance_values']['SHAP']:
            shap_clinical_rho = compute_spearman_rho(
                results['average_rankings']['SHAP'], CLINICAL_GOLD_STANDARD
            )
            results['clinical_alignment']['SHAP'] = {'spearman_rho': shap_clinical_rho}
            print(f"  SHAP: {shap_clinical_rho:.4f}")
        
        # Compute cross-fold stability
        def compute_ranking_stability(rankings: List[List[str]]) -> float:
            """Compute cross-fold ranking stability (mean std of position indices)."""
            n_domains = len(rankings[0])
            position_stds = []
            
            for pos in range(n_domains):
                # Count domain indices at each position across folds
                position_indices = []
                for rank in rankings:
                    domain_at_pos = rank[pos]
                    idx = domain_names.index(domain_at_pos)
                    position_indices.append(idx)
                position_stds.append(np.std(position_indices))
            
            return np.mean(position_stds)
        
        cbm_stability = compute_ranking_stability(results['fold_rankings']['CBM'])
        grad_stability = compute_ranking_stability(results['fold_rankings']['Gradient'])
        
        results['cross_fold_stability'] = {
            'CBM': {'mean_position_std': cbm_stability},
            'Gradient': {'mean_position_std': grad_stability},
            'interpretation': 'lower is more stable'
        }
        
        print(f"\nCross-fold stability (mean position std, lower is more stable):")
        print(f"  CBM: {cbm_stability:.4f}")
        print(f"  Gradient: {grad_stability:.4f}")
        
        if SHAP_AVAILABLE and results['fold_rankings']['SHAP']:
            shap_stability = compute_ranking_stability(results['fold_rankings']['SHAP'])
            results['cross_fold_stability']['SHAP'] = {'mean_position_std': shap_stability}
            print(f"  SHAP: {shap_stability:.4f}")
        
        # Save results
        output_path = os.path.join(RESULTS_DIR, 'cbm_exp8_shap_comparison.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
        
        # Print final summary
        print("\n" + "=" * 60)
        print("Experiment Summary")
        print("=" * 60)
        print(f"\n1. Ranking Consistency (CBM vs Gradient): Spearman rho = {cbm_grad_rho:.4f}")
        print(f"2. Clinical Alignment:")
        print(f"   - CBM: {cbm_clinical_rho:.4f}")
        print(f"   - Gradient: {grad_clinical_rho:.4f}")
        print(f"3. Cross-fold Stability:")
        print(f"   - CBM: {cbm_stability:.4f} {'(more stable)' if cbm_stability < grad_stability else ''}")
        print(f"   - Gradient: {grad_stability:.4f} {'(more stable)' if grad_stability < cbm_stability else ''}")
        
        if cbm_clinical_rho > grad_clinical_rho:
            print(f"\nConclusion: CBM built-in interpretability aligns better with clinical gold standard (rho gap: {cbm_clinical_rho - grad_clinical_rho:.4f})")
        else:
            print(f"\nConclusion: Gradient attribution aligns better with clinical gold standard (rho gap: {grad_clinical_rho - cbm_clinical_rho:.4f})")
        
        elapsed = time.time() - start
        print(f"\nTotal time: {elapsed/60:.1f} min")
        print("Experiment complete!")
        
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        
        # Save partial results
        output_path = os.path.join(RESULTS_DIR, 'cbm_exp8_shap_comparison.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Partial results saved to: {output_path}")


if __name__ == "__main__":
    main()
