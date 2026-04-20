"""
Experiment 6: Clinical alignment analysis.
Validates CBM predictions against clinical diagnostic criteria.

Validates the alignment between CBM concepts and clinical variables:
- Exp-12a: Concept-clinical score correlation
- Exp-12b: Clinical prior consistency validation
- Exp-12c: Concept discriminative power analysis

Key constraints:
- SEED = 42
- Windows num_workers = 0
- 5-fold cross-validation

Corresponds to Section 4.4 in the paper.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy.stats import pearsonr
from collections import OrderedDict

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score

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
    preprocess_with_concepts, compute_true_concepts,
    CBM_Supervised, train_cbm
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


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.
    d = (mean1 - mean2) / pooled_std
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (mean1 - mean2) / pooled_std


def extract_concepts_from_fold(model, X_test: np.ndarray, device: torch.device) -> np.ndarray:
    """Extract concept predictions from test set."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        _, concepts = model.predict_proba(X_tensor)
        return concepts.cpu().numpy()


def run_exp12a_correlation(X: np.ndarray, y: np.ndarray, dataset_name: str) -> Dict:
    """
    Exp-12a: Concept-clinical score correlation analysis.

    Computes Pearson correlation matrix between 7 concepts x 4 clinical variables.
    """
    print(f"\n[{dataset_name}] Exp-12a: Concept-Clinical Score Correlation")
    print("-" * 50)

    # Store per-fold results
    fold_corr_matrices = []
    fold_pvalue_matrices = []
    fold_n_samples = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold}...", end='')
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Preprocessing (including true concept computation)
        X_train_s, X_test_s, tc_train, tc_test = preprocess_with_concepts(X_train, X_test)

        # Train CBM-Supervised
        model = CBM_Supervised()
        res = train_cbm(
            model, X_train_s, y_train, X_test_s, y_test,
            concept_supervision=True,
            true_concepts_train=tc_train,
            true_concepts_test=tc_test,
            lambda_concept=0.5,
            lambda_polarity=0.1,
            verbose=False
        )
        
        # Extract concept predictions
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        concepts_pred = extract_concepts_from_fold(res['model'], X_test_s, device)
        
        # Extract imputed but unscaled clinical variables from original X
        imputer = KNNImputer(n_neighbors=5)
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        # Compute clinical variables
        moca_total = X_test_imp[:, :19].sum(axis=1)  # MoCA total score
        age = X_test_imp[:, 19]  # AGE
        education = X_test_imp[:, 21]  # EDUCATION
        gds = X_test_imp[:, 22]  # GDS
        
        clinical_vars = {
            'MoCA_total': moca_total,
            'AGE': age,
            'EDUCATION': education,
            'GDS': gds
        }
        
        # Compute Pearson correlation
        corr_matrix = {}
        pvalue_matrix = {}
        
        for i, domain in enumerate(COGNITIVE_DOMAINS.keys()):
            corr_matrix[domain] = {}
            pvalue_matrix[domain] = {}
            concept_values = concepts_pred[:, i]
            
            for var_name, var_values in clinical_vars.items():
                # Filter NaN
                valid_mask = ~(np.isnan(concept_values) | np.isnan(var_values))
                if valid_mask.sum() > 3:
                    r, p = pearsonr(concept_values[valid_mask], var_values[valid_mask])
                    corr_matrix[domain][var_name] = float(r)
                    pvalue_matrix[domain][var_name] = float(p)
                else:
                    corr_matrix[domain][var_name] = 0.0
                    pvalue_matrix[domain][var_name] = 1.0
        
        fold_corr_matrices.append(corr_matrix)
        fold_pvalue_matrices.append(pvalue_matrix)
        fold_n_samples.append(len(y_test))
        print(f" done (n={len(y_test)})")
    
    # Merge 5-fold results (take average)
    final_corr = {}
    final_pvalue = {}
    
    for domain in COGNITIVE_DOMAINS.keys():
        final_corr[domain] = {}
        final_pvalue[domain] = {}
        
        for var_name in ['MoCA_total', 'AGE', 'EDUCATION', 'GDS']:
            corrs = [fold[domain][var_name] for fold in fold_corr_matrices]
            pvalues = [fold[domain][var_name] for fold in fold_pvalue_matrices]
            final_corr[domain][var_name] = float(np.mean(corrs))
            final_pvalue[domain][var_name] = float(np.mean(pvalues))
    
    # Print results
    print(f"\n  Concept-Clinical Variable Pearson Correlation (5-fold average):")
    print(f"  {'Domain':<25} {'MoCA_total':>12} {'AGE':>10} {'EDUCATION':>12} {'GDS':>10}")
    print("  " + "-" * 75)
    for domain in COGNITIVE_DOMAINS.keys():
        print(f"  {domain:<25} "
              f"{final_corr[domain]['MoCA_total']:>12.3f} "
              f"{final_corr[domain]['AGE']:>10.3f} "
              f"{final_corr[domain]['EDUCATION']:>12.3f} "
              f"{final_corr[domain]['GDS']:>10.3f}")
    
    return {
        'correlation_matrix': final_corr,
        'p_value_matrix': final_pvalue,
        'n_folds': 5,
        'n_samples_per_fold': fold_n_samples
    }


def run_exp12b_clinical_priors(exp3_path: str, exp4_path: str, exp12a_results: Dict) -> Dict:
    """
    Exp-12b: Clinical prior consistency validation.

    Validates the following clinical priors:
    1. AD prior 1: Delayed_Recall is the most degraded domain from Normal to Dementia
    2. AD prior 2: Orientation degrades more in the MCI-to-Dementia stage
    3. Overall prior: All concepts positively correlated with MoCA total (r>0.5)
    4. PD prior: Visuospatial is among the more degraded domains in PD
    """
    print("\n[Exp-12b] Clinical Prior Consistency Validation")
    print("-" * 50)

    # Read existing result files
    with open(exp3_path, 'r', encoding='utf-8') as f:
        exp3_data = json.load(f)
    with open(exp4_path, 'r', encoding='utf-8') as f:
        exp4_data = json.load(f)
    
    prior_checks = []
    passed_count = 0

    # Prior 1: AD - Delayed_Recall is the most degraded domain from Normal to Dementia
    print("\n  Validating AD Prior 1: Delayed_Recall is the most degraded domain from Normal to Dementia")
    nacc_data = exp3_data.get('NACC', {})
    if 'classes' in nacc_data:
        normal_means = nacc_data['classes'].get('Normal', {}).get('concept_means', {})
        dementia_means = nacc_data['classes'].get('Dementia', {}).get('concept_means', {})
        
        diffs = {}
        for domain in COGNITIVE_DOMAINS.keys():
            if domain in normal_means and domain in dementia_means:
                diffs[domain] = normal_means[domain] - dementia_means[domain]
        
        if diffs:
            max_diff_domain = max(diffs, key=diffs.get)
            prior1_passed = (max_diff_domain == 'Delayed_Recall')
            prior1_evidence = f"Normal-Dementia diff: {diffs.get('Delayed_Recall', 0):.4f}"
            if diffs:
                prior1_evidence += f", largest among {len(diffs)} domains"
            
            prior_checks.append({
                'prior': 'AD: Delayed_Recall is most degraded domain (Normal->Dementia)',
                'result': prior1_passed,
                'evidence': prior1_evidence
            })
            if prior1_passed:
                passed_count += 1
            print(f"    Result: {'PASS' if prior1_passed else 'FAIL'} - {prior1_evidence}")

    # Prior 2: AD - Orientation degrades more in the MCI-to-Dementia stage
    print("\n  Validating AD Prior 2: Orientation degrades more in the MCI-to-Dementia stage")
    if 'classes' in nacc_data:
        normal_means = nacc_data['classes'].get('Normal', {}).get('concept_means', {})
        mci_means = nacc_data['classes'].get('MCI', {}).get('concept_means', {})
        dementia_means = nacc_data['classes'].get('Dementia', {}).get('concept_means', {})
        
        if all(k in normal_means for k in ['Orientation']) and \
           all(k in mci_means for k in ['Orientation']) and \
           all(k in dementia_means for k in ['Orientation']):
            diff_n_m = normal_means['Orientation'] - mci_means['Orientation']
            diff_m_d = mci_means['Orientation'] - dementia_means['Orientation']
            
            prior2_passed = (diff_m_d > diff_n_m)
            prior2_evidence = f"Normal->MCI diff={diff_n_m:.4f}, MCI->Dementia diff={diff_m_d:.4f}"
            
            prior_checks.append({
                'prior': 'AD: Orientation degrades more in MCI->Dementia than Normal->MCI',
                'result': prior2_passed,
                'evidence': prior2_evidence
            })
            if prior2_passed:
                passed_count += 1
            print(f"    Result: {'PASS' if prior2_passed else 'FAIL'} - {prior2_evidence}")

    # Prior 3: All concepts positively correlated with MoCA total (r>0.5)
    print("\n  Validating Overall Prior: All concepts positively correlated with MoCA total (r>0.5)")
    nacc_corr = exp12a_results.get('NACC', {}).get('correlation_matrix', {})
    ppmi_corr = exp12a_results.get('PPMI', {}).get('correlation_matrix', {})
    
    all_positive_nacc = all(
        nacc_corr.get(d, {}).get('MoCA_total', 0) > 0.5 
        for d in COGNITIVE_DOMAINS.keys()
    ) if nacc_corr else False
    
    all_positive_ppmi = all(
        ppmi_corr.get(d, {}).get('MoCA_total', 0) > 0.5 
        for d in COGNITIVE_DOMAINS.keys()
    ) if ppmi_corr else False
    
    min_r_nacc = min(
        nacc_corr.get(d, {}).get('MoCA_total', 0) 
        for d in COGNITIVE_DOMAINS.keys()
    ) if nacc_corr else 0
    min_r_ppmi = min(
        ppmi_corr.get(d, {}).get('MoCA_total', 0) 
        for d in COGNITIVE_DOMAINS.keys()
    ) if ppmi_corr else 0
    
    prior3_passed = all_positive_nacc and all_positive_ppmi
    prior3_evidence = f"NACC min r={min_r_nacc:.3f}, PPMI min r={min_r_ppmi:.3f}"
    
    prior_checks.append({
        'prior': 'All concepts positively correlated with MoCA total (r>0.5)',
        'result': prior3_passed,
        'evidence': prior3_evidence
    })
    if prior3_passed:
        passed_count += 1
    print(f"    Result: {'PASS' if prior3_passed else 'FAIL'} - {prior3_evidence}")

    # Prior 4: PD - Visuospatial is among the more degraded domains in PD
    print("\n  Validating PD Prior: Visuospatial is among the more degraded domains in PD")
    ppmi_data = exp3_data.get('PPMI', {})
    if 'classes' in ppmi_data:
        normal_means = ppmi_data['classes'].get('Normal', {}).get('concept_means', {})
        dementia_means = ppmi_data['classes'].get('Dementia', {}).get('concept_means', {})
        
        diffs = {}
        for domain in COGNITIVE_DOMAINS.keys():
            if domain in normal_means and domain in dementia_means:
                diffs[domain] = normal_means[domain] - dementia_means[domain]
        
        if diffs:
            # Sort; check whether Visuospatial is in the top 3
            sorted_domains = sorted(diffs.items(), key=lambda x: x[1], reverse=True)
            visuo_rank = next(i for i, (d, _) in enumerate(sorted_domains) if d == 'Visuospatial_Executive') + 1

            prior4_passed = (visuo_rank <= 3)  # Top 3 counts as pass
            prior4_evidence = f"Visuospatial rank #{visuo_rank} in degradation (diff={diffs.get('Visuospatial_Executive', 0):.4f})"
            
            prior_checks.append({
                'prior': 'PD: Visuospatial_Executive is among top degraded domains',
                'result': prior4_passed,
                'evidence': prior4_evidence
            })
            if prior4_passed:
                passed_count += 1
            print(f"    Result: {'PASS' if prior4_passed else 'FAIL'} - {prior4_evidence}")

    # Compute clinical consistency index
    total_priors = len(prior_checks)
    consistency_index = passed_count / total_priors if total_priors > 0 else 0

    print(f"\n  Clinical consistency index: {consistency_index:.2f} ({passed_count}/{total_priors})")
    
    return {
        'prior_checks': prior_checks,
        'clinical_consistency_index': consistency_index
    }


def run_exp12c_discriminative_power(X: np.ndarray, y: np.ndarray, dataset_name: str) -> Dict:
    """
    Exp-12c: Concept discriminative power analysis.

    Computes Cohen's d and ROC-AUC for each concept.
    """
    print(f"\n[{dataset_name}] Exp-12c: Concept Discriminative Power Analysis")
    print("-" * 50)

    # Store per-fold results
    fold_results = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold}...", end='')
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Preprocessing
        X_train_s, X_test_s, tc_train, tc_test = preprocess_with_concepts(X_train, X_test)

        # Train CBM-Supervised
        model = CBM_Supervised()
        res = train_cbm(
            model, X_train_s, y_train, X_test_s, y_test,
            concept_supervision=True,
            true_concepts_train=tc_train,
            true_concepts_test=tc_test,
            lambda_concept=0.5,
            lambda_polarity=0.1
        )
        
        # Extract concept predictions
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        concepts_pred = extract_concepts_from_fold(res['model'], X_test_s, device)
        
        fold_results.append({
            'concepts': concepts_pred,
            'y': y_test
        })
        print(" done")
    
    # Merge all folds
    all_concepts = np.vstack([f['concepts'] for f in fold_results])
    all_y = np.concatenate([f['y'] for f in fold_results])

    # Compute discriminative power metrics
    normal_vs_mci = []
    mci_vs_dementia = []

    # Compute per concept
    for i, domain in enumerate(COGNITIVE_DOMAINS.keys()):
        concept_values = all_concepts[:, i]
        
        # Normal vs MCI
        normal_mask = all_y == 0
        mci_mask = all_y == 1
        dementia_mask = all_y == 2
        
        if normal_mask.sum() > 0 and mci_mask.sum() > 0:
            d_nm = compute_cohens_d(
                concept_values[normal_mask],
                concept_values[mci_mask]
            )
            
            # ROC-AUC for Normal vs Non-Normal
            y_binary = (all_y > 0).astype(int)  # 0=Normal, 1=Non-Normal
            try:
                auc = roc_auc_score(y_binary, -concept_values)  # Negative sign because lower concept value = more likely Non-Normal
            except:
                auc = 0.5
            
            normal_vs_mci.append({
                'domain': domain,
                'cohens_d': float(d_nm),
                'roc_auc': float(auc)
            })
        
        # MCI vs Dementia
        if mci_mask.sum() > 0 and dementia_mask.sum() > 0:
            d_md = compute_cohens_d(
                concept_values[mci_mask],
                concept_values[dementia_mask]
            )
            
            # ROC-AUC for MCI vs Dementia
            mask_md = (all_y == 1) | (all_y == 2)
            y_binary_md = (all_y[mask_md] == 2).astype(int)  # 0=MCI, 1=Dementia
            try:
                auc_md = roc_auc_score(y_binary_md, -concept_values[mask_md])
            except:
                auc_md = 0.5
            
            mci_vs_dementia.append({
                'domain': domain,
                'cohens_d': float(d_md),
                'roc_auc': float(auc_md)
            })
    
    # Sort by Cohen's d
    normal_vs_mci_sorted = sorted(normal_vs_mci, key=lambda x: x['cohens_d'], reverse=True)
    mci_vs_dementia_sorted = sorted(mci_vs_dementia, key=lambda x: x['cohens_d'], reverse=True)

    # Print results
    print("\n  Normal vs MCI (sorted by Cohen's d):")
    print(f"  {'Domain':<25} {'Cohens d':>12} {'ROC-AUC':>10}")
    print("  " + "-" * 50)
    for item in normal_vs_mci_sorted:
        print(f"  {item['domain']:<25} {item['cohens_d']:>12.3f} {item['roc_auc']:>10.3f}")
    
    print("\n  MCI vs Dementia (sorted by Cohen's d):")
    print(f"  {'Domain':<25} {'Cohens d':>12} {'ROC-AUC':>10}")
    print("  " + "-" * 50)
    for item in mci_vs_dementia_sorted:
        print(f"  {item['domain']:<25} {item['cohens_d']:>12.3f} {item['roc_auc']:>10.3f}")
    
    return {
        'normal_vs_mci': normal_vs_mci_sorted,
        'mci_vs_dementia': mci_vs_dementia_sorted
    }


def main():
    print("=" * 60)
    print("CBM Clinical Alignment Validation Experiment (Exp-12)")
    print("=" * 60)
    print(f"Start time: {pd.Timestamp.now()}")

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
        
        # ========== Exp-12a: Concept-clinical score correlation ==========
        exp12a_nacc = run_exp12a_correlation(X_nacc, y_nacc, 'NACC')
        exp12a_ppmi = run_exp12a_correlation(X_ppmi, y_ppmi, 'PPMI')
        
        exp12a_results = {
            'NACC': exp12a_nacc,
            'PPMI': exp12a_ppmi
        }
        
        # ========== Exp-12b: Clinical prior consistency validation ==========
        exp3_path = os.path.join(RESULTS_DIR, 'cbm_exp3_interpretability.json')
        exp4_path = os.path.join(RESULTS_DIR, 'cbm_exp4_counterfactual.json')
        
        if os.path.exists(exp3_path) and os.path.exists(exp4_path):
            exp12b_results = run_exp12b_clinical_priors(exp3_path, exp4_path, exp12a_results)
        else:
            print(f"\n[Warning] Missing required result files, skipping Exp-12b")
            print(f"  Required: {exp3_path}")
            print(f"  Required: {exp4_path}")
            exp12b_results = {
                'prior_checks': [],
                'clinical_consistency_index': 0.0,
                'error': 'Missing required result files'
            }
        
        # ========== Exp-12c: Concept discriminative power analysis ==========
        exp12c_nacc = run_exp12c_discriminative_power(X_nacc, y_nacc, 'NACC')
        exp12c_ppmi = run_exp12c_discriminative_power(X_ppmi, y_ppmi, 'PPMI')
        
        exp12c_results = {
            'NACC': exp12c_nacc,
            'PPMI': exp12c_ppmi
        }
        
        # ========== Save results ==========
        final_results = {
            'exp12a_correlation': exp12a_results,
            'exp12b_clinical_priors': exp12b_results,
            'exp12c_discriminative_power': exp12c_results
        }
        
        output_path = os.path.join(RESULTS_DIR, 'cbm_exp12_clinical_alignment.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'=' * 60}")
        print(f"Results saved to: {output_path}")
        print(f"{'=' * 60}")

        # Print summary
        print("\n[Experiment Summary]")
        print("-" * 50)

        # Exp-12a summary
        print("\nExp-12a: Concept-MoCA Total Correlation (5-fold average)")
        for ds_name in ['NACC', 'PPMI']:
            corr_data = exp12a_results[ds_name]['correlation_matrix']
            mean_r = np.mean([corr_data[d]['MoCA_total'] for d in COGNITIVE_DOMAINS.keys()])
            print(f"  {ds_name}: mean r = {mean_r:.3f}")

        # Exp-12b summary
        if 'clinical_consistency_index' in exp12b_results:
            print(f"\nExp-12b: Clinical consistency index = {exp12b_results['clinical_consistency_index']:.2f}")
            print("  Prior validation results:")
            for check in exp12b_results.get('prior_checks', []):
                status = "PASS" if check['result'] else "FAIL"
                print(f"    [{status}] {check['prior']}")

        # Exp-12c summary
        print("\nExp-12c: Concept Discriminative Power (highest Cohen's d domain)")
        for ds_name in ['NACC', 'PPMI']:
            top_nm = exp12c_results[ds_name]['normal_vs_mci'][0]
            top_md = exp12c_results[ds_name]['mci_vs_dementia'][0]
            print(f"  {ds_name}:")
            print(f"    Normal vs MCI: {top_nm['domain']} (d={top_nm['cohens_d']:.3f})")
            print(f"    MCI vs Dementia: {top_md['domain']} (d={top_md['cohens_d']:.3f})")

        print(f"\n{'=' * 60}")
        print("CBM Clinical Alignment Validation Experiment Completed!")
        print(f"{'=' * 60}")
        
    except Exception as e:
        import traceback
        print(f"\n[Error] {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
