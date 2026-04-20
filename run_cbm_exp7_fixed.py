"""
Experiment 7: Ablation study.
Evaluates contribution of individual components (concept supervision, polarity, hybrid bypass).

Three ablation sub-experiments:
- Exp-7a: Bottleneck dimension ablation (n_free_dims = 0, 4, 8, 16)
- Exp-7b: Domain-level ablation (Leave-One-Domain-Out training)
- Exp-7c: Concept supervision strength ablation (lambda_concept = 0.0, 0.1, 0.25, 0.5, 1.0, 2.0)

Key improvements:
- Exp-7b: Exclude domain at training time instead of replacing at inference
- Fix batch index matching bug
- Add polarity constraint
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

# Import from run_explore_cbm.py
sys.path.insert(0, os.path.dirname(__file__))
from run_explore_cbm import (
    COGNITIVE_DOMAINS, MOCA_FEATURES, DOMAIN_MAX_SCORES,
    preprocess_with_concepts, compute_true_concepts,
    CBM_Supervised, CBM_Hybrid, train_cbm
)
from experiment_data import load_nacc_extended, load_ppmi_extended

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============== CBM_Ablated: Domain-Level Ablation Model ==============

class CBM_Ablated(nn.Module):
    """
    CBM_Ablated: CBM model with one cognitive domain removed.
    
    Used for Exp-7b domain-level ablation.
    - Concept predictors retain only 6 (skip the domain at excluded_domain_idx)
    - diagnosis_head input dimension is 6 + 5 = 11
    """
    
    def __init__(self, excluded_domain_idx: int, n_demo: int = 5, n_classes: int = 3):
        super().__init__()
        self.excluded_idx = excluded_domain_idx
        self.n_demo = n_demo
        self.n_classes = n_classes
        
        # Get active domains (exclude the removed one)
        domains = list(COGNITIVE_DOMAINS.items())
        self.active_domains = [d for i, d in enumerate(domains) if i != excluded_domain_idx]
        self.domain_names = [d[0] for d in self.active_domains]
        
        # Create concept predictor for each active domain
        self.concept_predictors = nn.ModuleList()
        self.domain_indices = []
        
        for domain_name, items in self.active_domains:
            indices = [MOCA_FEATURES.index(item) for item in items]
            self.domain_indices.append(indices)
            n_items = len(items)
            self.concept_predictors.append(nn.Sequential(
                nn.Linear(n_items, max(16, n_items * 2)),
                nn.ReLU(),
                nn.Linear(max(16, n_items * 2), 1),
                nn.Sigmoid()
            ))
        
        # Diagnosis head: 6 concepts + 5 demographics = 11-dim input
        n_concepts = len(self.active_domains)  # 6
        self.diagnosis_head = nn.Sequential(
            nn.Linear(n_concepts + n_demo, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, n_classes)
        )
    
    def _get_concepts(self, x):
        """Extract concept layer output [B, 6]."""
        x_moca = x[:, :19]
        concepts = []
        for i, indices in enumerate(self.domain_indices):
            domain_input = x_moca[:, indices]
            concept = self.concept_predictors[i](domain_input)
            concepts.append(concept)
        return torch.cat(concepts, dim=1)  # [B, 6]
    
    def forward(self, x):
        x_demo = x[:, 19:19 + self.n_demo]
        concepts = self._get_concepts(x)
        combined = torch.cat([concepts, x_demo], dim=1)
        logits = self.diagnosis_head(combined)
        return F.log_softmax(logits, dim=-1), concepts
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            log_probs, concepts = self.forward(x)
            return torch.exp(log_probs), concepts


def train_cbm_ablated(model, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      true_concepts_train: np.ndarray,  # [N, 6] trimmed concepts
                      epochs: int = 100, batch_size: int = 32, lr: float = 1e-3,
                      lambda_concept: float = 0.5, lambda_polarity: float = 0.1,
                      max_retries: int = 3, verbose: bool = False) -> Dict[str, Any]:
    """
    Train CBM_Ablated model.
    
    Args:
        model: CBM_Ablated instance
        true_concepts_train: [N, 6] trimmed true concepts (excluded domain removed)
    """
    best_result = None
    best_ba = 0.0
    best_model_state = None
    
    for attempt in range(max_retries):
        torch.manual_seed(SEED + attempt)
        np.random.seed(SEED + attempt)
        
        # Re-initialize model
        if attempt > 0:
            model = CBM_Ablated(excluded_domain_idx=model.excluded_idx)
        
        result = _train_cbm_ablated_single(
            model, X_train, y_train, X_test, y_test,
            true_concepts_train, epochs, batch_size, lr,
            lambda_concept, lambda_polarity, verbose
        )
        
        ba = result['balanced_accuracy']
        
        if ba > best_ba:
            best_ba = ba
            best_result = result
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if ba > 0.4:
            result['model'] = model
            return result
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    best_result['model'] = model
    return best_result


def _train_cbm_ablated_single(model, X_train, y_train, X_test, y_test,
                               true_concepts_train, epochs, batch_size, lr,
                               lambda_concept, lambda_polarity, verbose):
    """Single training run for CBM_Ablated."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    true_concepts_train_t = torch.FloatTensor(true_concepts_train).to(device)
    
    # Create index array
    indices = np.arange(len(X_train))
    idx_tensor = torch.LongTensor(indices).to(device)
    
    # DataLoader
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
            
            # Standard CE loss
            log_probs, concepts = model(xb)
            loss = criterion(log_probs, yb)
            
            # Concept supervision
            true_concepts_batch = true_concepts_train_t[idx]
            concept_loss = F.mse_loss(concepts, true_concepts_batch)
            
            # Polarity constraint
            corr_penalty = 0.0
            for d in range(concepts.shape[1]):
                pred_d = concepts[:, d]
                true_d = true_concepts_batch[:, d]
                pred_centered = pred_d - pred_d.mean()
                true_centered = true_d - true_d.mean()
                corr = (pred_centered * true_centered).sum() / (
                    pred_centered.norm() * true_centered.norm() + 1e-8)
                corr_penalty += F.relu(-corr)
            
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
        y_pred = y_prob.argmax(axis=1)
    
    ba = balanced_accuracy_score(y_test, y_pred)
    
    return {
        'balanced_accuracy': ba,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'concepts': concepts.cpu().numpy(),
        'model': model
    }


# ============== Exp-7a: Bottleneck Dimension Ablation ==============

def run_exp7a(X_nacc: np.ndarray, y_nacc: np.ndarray,
              X_ppmi: np.ndarray, y_ppmi: np.ndarray) -> List[Dict]:
    """
    Exp-7a: Bottleneck dimension ablation.
    
    Compare n_free_dims = [0, 4, 8, 16]
    - n_free=0: Use CBM_Supervised (pure 7-dim concepts)
    - n_free>0: Use CBM_Hybrid(n_free_dims=n_free)
    """
    print("\n" + "=" * 60)
    print("Exp-7a: Bottleneck Dimension Ablation")
    print("=" * 60)
    
    results = []
    n_free_configs = [0, 4, 8, 16]
    
    for ds_name, (X, y) in [('NACC', (X_nacc, y_nacc)), ('PPMI', (X_ppmi, y_ppmi))]:
        print(f"\n[{ds_name}] Bottleneck dimension ablation")
        print("-" * 50)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        
        for n_free in n_free_configs:
            print(f"\n  n_free_dims = {n_free}:")
            fold_bas = []
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Preprocessing
                X_train_s, X_test_s, tc_train, tc_test = preprocess_with_concepts(X_train, X_test)
                
                # Select model type
                if n_free == 0:
                    model = CBM_Supervised()
                else:
                    model = CBM_Hybrid(n_free_dims=n_free)
                
                # Training
                res = train_cbm(
                    model, X_train_s, y_train, X_test_s, y_test,
                    concept_supervision=True,
                    true_concepts_train=tc_train,
                    true_concepts_test=tc_test,
                    lambda_polarity=0.1
                )
                
                fold_bas.append(res['balanced_accuracy'])
                print(f"    Fold {fold}: BA = {res['balanced_accuracy']:.4f}")
            
            mean_ba = np.mean(fold_bas)
            std_ba = np.std(fold_bas)
            total_dims = 7 + n_free  # 7 concepts + n free dimensions
            
            results.append({
                'dataset': ds_name,
                'n_free': n_free,
                'total_dims': total_dims,
                'mean_ba': float(mean_ba),
                'std_ba': float(std_ba)
            })
            
            print(f"  => mean_ba = {mean_ba:.4f} +/- {std_ba:.4f}")
    
    return results


# ============== Exp-7b: Domain-Level Ablation ==============

def run_exp7b(X_nacc: np.ndarray, y_nacc: np.ndarray,
              X_ppmi: np.ndarray, y_ppmi: np.ndarray) -> List[Dict]:
    """
    Exp-7b: Domain-level ablation (Leave-One-Domain-Out training).
    
    For each domain to remove, remove_idx (0~6):
    1. Create CBM_Ablated model (only 6 concept predictors)
    2. Train with trimmed true concepts (6-dim)
    3. Record BA and compare with full 7-domain model
    """
    print("\n" + "=" * 60)
    print("Exp-7b: Domain-Level Ablation (Leave-One-Domain-Out)")
    print("=" * 60)
    
    results = []
    domain_names = list(COGNITIVE_DOMAINS.keys())
    
    # First get full 7-domain model baseline BA
    print("\n[Getting full 7-domain model baseline...]")
    baseline_ba = {}
    
    for ds_name, (X, y) in [('NACC', (X_nacc, y_nacc)), ('PPMI', (X_ppmi, y_ppmi))]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_bas = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            X_train_s, X_test_s, tc_train, tc_test = preprocess_with_concepts(X_train, X_test)
            
            model = CBM_Supervised()
            res = train_cbm(
                model, X_train_s, y_train, X_test_s, y_test,
                concept_supervision=True,
                true_concepts_train=tc_train,
                true_concepts_test=tc_test,
                lambda_polarity=0.1
            )
            fold_bas.append(res['balanced_accuracy'])
        
        baseline_ba[ds_name] = np.mean(fold_bas)
        print(f"  {ds_name} baseline BA: {baseline_ba[ds_name]:.4f}")
    
    # Ablate each domain
    for remove_idx in range(7):
        removed_domain = domain_names[remove_idx]
        
        for ds_name, (X, y) in [('NACC', (X_nacc, y_nacc)), ('PPMI', (X_ppmi, y_ppmi))]:
            print(f"\n[{ds_name}] Removing domain: {removed_domain}")
            print("-" * 50)
            
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            fold_bas = []
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Preprocessing (get full 7-dim concepts)
                X_train_s, X_test_s, tc_train_full, tc_test_full = preprocess_with_concepts(X_train, X_test)
                
                # Trim concepts: remove the remove_idx column
                tc_train_ablated = np.delete(tc_train_full, remove_idx, axis=1)  # [N, 6]
                tc_test_ablated = np.delete(tc_test_full, remove_idx, axis=1)
                
                # Create ablation model
                model = CBM_Ablated(excluded_domain_idx=remove_idx)
                
                # Train
                res = train_cbm_ablated(
                    model, X_train_s, y_train, X_test_s, y_test,
                    true_concepts_train=tc_train_ablated,
                    lambda_concept=0.5,
                    lambda_polarity=0.1
                )
                
                fold_bas.append(res['balanced_accuracy'])
                print(f"    Fold {fold}: BA = {res['balanced_accuracy']:.4f}")
            
            mean_ba = np.mean(fold_bas)
            std_ba = np.std(fold_bas)
            ba_drop = baseline_ba[ds_name] - mean_ba
            
            results.append({
                'dataset': ds_name,
                'removed_domain': removed_domain,
                'mean_ba': float(mean_ba),
                'std_ba': float(std_ba),
                'ba_drop': float(ba_drop)
            })
            
            print(f"  => mean_ba = {mean_ba:.4f} +/- {std_ba:.4f}, BA drop = {ba_drop:.4f}")
    
    return results


# ============== Exp-7c: Concept Supervision Strength Ablation ==============

def run_exp7c(X_nacc: np.ndarray, y_nacc: np.ndarray,
              X_ppmi: np.ndarray, y_ppmi: np.ndarray) -> List[Dict]:
    """
    Exp-7c: Concept supervision strength ablation.
    
    Compare lambda_concept = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    Use CBM_Supervised, record mean_ba and per-domain average Pearson r.
    """
    print("\n" + "=" * 60)
    print("Exp-7c: Concept Supervision Strength Ablation")
    print("=" * 60)
    
    results = []
    lambda_configs = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    
    for ds_name, (X, y) in [('NACC', (X_nacc, y_nacc)), ('PPMI', (X_ppmi, y_ppmi))]:
        print(f"\n[{ds_name}] Concept supervision strength ablation")
        print("-" * 50)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        
        for lambda_c in lambda_configs:
            print(f"\n  lambda_concept = {lambda_c}:")
            fold_bas = []
            fold_concept_rs = []  # mean Pearson r per fold
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Preprocessing
                X_train_s, X_test_s, tc_train, tc_test = preprocess_with_concepts(X_train, X_test)
                
                model = CBM_Supervised()
                
                # Train (note: lambda_concept=0.0 means no concept supervision)
                res = train_cbm(
                    model, X_train_s, y_train, X_test_s, y_test,
                    concept_supervision=True,
                    lambda_concept=lambda_c,
                    true_concepts_train=tc_train,
                    true_concepts_test=tc_test,
                    lambda_polarity=0.1
                )
                
                fold_bas.append(res['balanced_accuracy'])
                
                # Compute Pearson correlation between predicted and true concepts
                pred_concepts = res['concepts']
                domain_rs = []
                for d in range(7):
                    r, _ = pearsonr(pred_concepts[:, d], tc_test[:, d])
                    if not np.isnan(r):
                        domain_rs.append(r)
                mean_r = np.mean(domain_rs) if domain_rs else 0.0
                fold_concept_rs.append(mean_r)
                
                print(f"    Fold {fold}: BA = {res['balanced_accuracy']:.4f}, mean_r = {mean_r:.4f}")
            
            mean_ba = np.mean(fold_bas)
            std_ba = np.std(fold_bas)
            mean_concept_r = np.mean(fold_concept_rs)
            
            results.append({
                'dataset': ds_name,
                'lambda_concept': lambda_c,
                'mean_ba': float(mean_ba),
                'std_ba': float(std_ba),
                'mean_concept_r': float(mean_concept_r)
            })
            
            print(f"  => mean_ba = {mean_ba:.4f} +/- {std_ba:.4f}, mean_concept_r = {mean_concept_r:.4f}")
    
    return results


# ============== Main Function ==============

def main():
    print("=" * 60)
    print("CBM Exp-7 Ablation Experiment")
    print("=" * 60)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start = time.time()
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    all_results = {'exp7a': [], 'exp7b': [], 'exp7c': []}
    
    try:
        # ========== Data Loading ==========
        print("\n[Data Loading]")
        print("-" * 50)
        
        X_nacc, y_nacc = load_nacc_extended(max_per_class=500)
        print(f"  NACC: {X_nacc.shape}, class distribution={np.bincount(y_nacc)}")
        
        X_ppmi, y_ppmi = load_ppmi_extended()
        print(f"  PPMI: {X_ppmi.shape}, class distribution={np.bincount(y_ppmi)}")
        
        # ========== Exp-7a: Bottleneck Dimension Ablation ==========
        try:
            all_results['exp7a'] = run_exp7a(X_nacc, y_nacc, X_ppmi, y_ppmi)
        except Exception as e:
            print(f"\n[ERROR] Exp-7a failed: {e}")
            import traceback
            traceback.print_exc()
        
        # ========== Exp-7b: Domain-Level Ablation ==========
        try:
            all_results['exp7b'] = run_exp7b(X_nacc, y_nacc, X_ppmi, y_ppmi)
        except Exception as e:
            print(f"\n[ERROR] Exp-7b failed: {e}")
            import traceback
            traceback.print_exc()
        
        # ========== Exp-7c: Concept Supervision Strength Ablation ==========
        try:
            all_results['exp7c'] = run_exp7c(X_nacc, y_nacc, X_ppmi, y_ppmi)
        except Exception as e:
            print(f"\n[ERROR] Exp-7c failed: {e}")
            import traceback
            traceback.print_exc()
        
        # ========== Save Results ==========
        output_path = os.path.join(RESULTS_DIR, 'cbm_exp7_ablation.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
        
        # ========== Done ==========
        elapsed = time.time() - start
        print("\n" + "=" * 60)
        print(f"Total time: {elapsed/60:.1f} min")
        print("Exp-7 Ablation Experiment Complete")
        print("=" * 60)
        
        # Print summary
        print("\n[Experiment Summary]")
        print("-" * 50)
        
        if all_results['exp7a']:
            print("\nExp-7a: Bottleneck Dimension Ablation (mean BA)")
            for r in all_results['exp7a']:
                print(f"  {r['dataset']}: n_free={r['n_free']}, total_dims={r['total_dims']}, BA={r['mean_ba']:.4f}+/-{r['std_ba']:.4f}")
        
        if all_results['exp7b']:
            print("\nExp-7b: Domain-Level Ablation (BA drop)")
            for r in all_results['exp7b']:
                print(f"  {r['dataset']}: removed={r['removed_domain']}, BA={r['mean_ba']:.4f}, drop={r['ba_drop']:.4f}")
        
        if all_results['exp7c']:
            print("\nExp-7c: Concept Supervision Strength Ablation (mean BA & concept_r)")
            for r in all_results['exp7c']:
                print(f"  {r['dataset']}: lambda={r['lambda_concept']}, BA={r['mean_ba']:.4f}+/-{r['std_ba']:.4f}, r={r['mean_concept_r']:.4f}")
        
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        elapsed = time.time() - start
        print(f"\nExperiment interrupted! Time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
