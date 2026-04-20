"""
Experiment 9: Edge computing deployment analysis.
Estimates INT8 quantization model size and inference latency on ESP32-S3.
"""
import os
import sys
import json
import time
import pickle
import struct
import warnings
import numpy as np
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARNING] xgboost not installed, skipping XGBoost analysis")

# ========== Cognitive Domain Definitions (consistent with run_explore_cbm.py) ==========
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

# Total input dimension: 19 MoCA + 5 demographics = 24
N_DEMO = 5
N_MOCA = 19
N_INPUT = N_MOCA + N_DEMO  # 24-dim (MoCA + demographics)
N_CONCEPTS = 7
N_CLASSES = 3
N_HIDDEN = 16  # diagnosis head hidden layer
N_FREE_DIMS = 8  # Hybrid free dimensions


# ========== CBM Model Definitions ==========

class CBM_Supervised(nn.Module):
    """
    CBM-Supervised: Concept Bottleneck Model with concept supervision.
    Architecture: 24-dim input -> 7 domain concept predictors -> concept layer(7) + demographics(5) -> diagnosis head -> 3 classes
    Diagnosis head: Linear(12, 16) -> BN -> ReLU -> Dropout -> Linear(16, 3)
    """

    def __init__(self, n_demo=N_DEMO, n_classes=N_CLASSES):
        super().__init__()
        self.domain_names = list(COGNITIVE_DOMAINS.keys())
        self.domain_indices = []
        self.n_demo = n_demo
        self.n_classes = n_classes

        # 7 concept predictors
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

        # Diagnosis head: 7 concepts + 5 demographics = 12 -> 16 -> 3
        n_concepts = len(COGNITIVE_DOMAINS)
        self.diagnosis_head = nn.Sequential(
            nn.Linear(n_concepts + n_demo, N_HIDDEN),
            nn.BatchNorm1d(N_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(N_HIDDEN, n_classes)
        )

    def forward(self, x):
        x_moca = x[:, :N_MOCA]
        x_demo = x[:, N_MOCA:N_MOCA + self.n_demo]
        concepts = []
        for i, indices in enumerate(self.domain_indices):
            domain_input = x_moca[:, indices]
            concept = self.concept_predictors[i](domain_input)
            concepts.append(concept)
        concepts = torch.cat(concepts, dim=1)
        combined = torch.cat([concepts, x_demo], dim=1)
        logits = self.diagnosis_head(combined)
        return F.log_softmax(logits, dim=-1), concepts


class CBM_Hybrid(nn.Module):
    """
    CBM-Hybrid: 7 concepts + 8 free latent dimensions.
    Architecture: 24-dim input -> 7 concept predictors + free encoder -> diagnosis head -> 3 classes
    """

    def __init__(self, n_demo=N_DEMO, n_classes=N_CLASSES, n_free_dims=N_FREE_DIMS):
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

        # Free encoder: 24-dim -> 32 -> 8
        self.free_encoder = nn.Sequential(
            nn.Linear(N_INPUT, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_free_dims)
        )

        # Diagnosis head: 7 concepts + 8 free + 5 demographics = 20 -> 16 -> 3
        n_concepts = len(COGNITIVE_DOMAINS)
        self.diagnosis_head = nn.Sequential(
            nn.Linear(n_concepts + n_free_dims + n_demo, N_HIDDEN),
            nn.BatchNorm1d(N_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(N_HIDDEN, n_classes)
        )

    def forward(self, x):
        x_moca = x[:, :N_MOCA]
        x_demo = x[:, N_MOCA:N_MOCA + self.n_demo]
        concepts = []
        for i, indices in enumerate(self.domain_indices):
            domain_input = x_moca[:, indices]
            concept = self.concept_predictors[i](domain_input)
            concepts.append(concept)
        concepts = torch.cat(concepts, dim=1)
        free = self.free_encoder(x)
        combined = torch.cat([concepts, free, x_demo], dim=1)
        logits = self.diagnosis_head(combined)
        return F.log_softmax(logits, dim=-1), concepts


# ========== Data Loading ==========

def load_data():
    """Load NACC 3-class data."""
    try:
        from experiment_data import load_nacc_ternary, preprocess_data
        X, y = load_nacc_ternary()
        X, y = preprocess_data(X, y, use_smote=False)
        print(f"[Data] NACC 3-class data loaded: X={X.shape}, y={y.shape}")
        return X, y
    except Exception as e:
        print(f"[Data] load_nacc_ternary failed: {e}")
        print("[Data] Using synthetic data instead")
        np.random.seed(42)
        X = np.random.randn(300, N_INPUT).astype(np.float32)
        y = np.random.randint(0, 3, 300)
        return X, y


# ========== Parameter Counting ==========

def count_torch_params(model):
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_lr_params(lr_model):
    """LR parameter count: coef + intercept."""
    n = lr_model.coef_.size + lr_model.intercept_.size
    return int(n)


def count_svm_params(svm_model):
    """SVM parameter count: n_support_vectors * n_features + intercept."""
    try:
        n = svm_model.support_vectors_.size + svm_model.intercept_.size
        return int(n)
    except Exception:
        return 0


def count_rf_params(rf_model):
    """RF parameter count: estimate total nodes across all trees (3 values per node)."""
    total_nodes = sum(est.tree_.node_count for est in rf_model.estimators_)
    return int(total_nodes * 3)


def count_xgb_params(xgb_model):
    """XGBoost parameter count: estimate nodes via tree dump."""
    try:
        dump = xgb_model.get_booster().get_dump()
        total_lines = sum(len(tree.split('\n')) for tree in dump)
        return int(total_lines)
    except Exception:
        return 0


def count_mlp_params(mlp_model):
    """MLP parameter count: weights + biases."""
    n = sum(w.size for w in mlp_model.coefs_) + sum(b.size for b in mlp_model.intercepts_)
    return int(n)


# ========== Model Size ==========

def get_torch_model_size_kb(model):
    """PyTorch model FP32 size (KB)."""
    n_params = count_torch_params(model)
    return round(n_params * 4 / 1024, 2)  # 4 bytes per float32


def get_sklearn_model_size_kb(model):
    """sklearn model pickle size (KB)."""
    data = pickle.dumps(model)
    return round(len(data) / 1024, 2)


# ========== Inference Latency ==========

def measure_torch_inference_ms(model, x_sample, n_repeats=1000):
    """Measure single-sample CPU inference latency for PyTorch model (ms)."""
    model.eval()
    device = torch.device('cpu')
    model = model.to(device)
    x_tensor = torch.FloatTensor(x_sample.reshape(1, -1)).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(50):
            _ = model(x_tensor)

    # Formal timing
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_repeats):
            _ = model(x_tensor)
    end = time.perf_counter()

    avg_ms = (end - start) / n_repeats * 1000
    return round(avg_ms, 4)


def measure_sklearn_inference_ms(model, x_sample, n_repeats=1000):
    """Measure single-sample CPU inference latency for sklearn model (ms)."""
    x = x_sample.reshape(1, -1)

    # Warmup
    for _ in range(50):
        _ = model.predict(x)

    start = time.perf_counter()
    for _ in range(n_repeats):
        _ = model.predict(x)
    end = time.perf_counter()

    avg_ms = (end - start) / n_repeats * 1000
    return round(avg_ms, 4)


# ========== Memory Footprint Estimation ==========

def estimate_torch_memory_kb(model, x_sample):
    """
    Estimate PyTorch model runtime memory (KB)
    = model parameters + activation values + I/O buffers
    """
    # Parameter memory
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    # Activation estimate: intermediate layer outputs, roughly 50% of parameter memory
    activation_bytes = param_bytes * 0.5
    # I/O buffers
    io_bytes = x_sample.nbytes * 2
    total_bytes = param_bytes + activation_bytes + io_bytes
    return round(total_bytes / 1024, 2)


def estimate_sklearn_memory_kb(model):
    """Estimate sklearn model memory (KB), based on pickle size x 1.5 factor."""
    pkl_size = len(pickle.dumps(model))
    return round(pkl_size * 1.5 / 1024, 2)


# ========== Edge Device Feasibility Analysis ==========

# Reference baseline: Intel i7-10th Gen single-core ~2000 MFLOPS
REF_FREQ_MHZ = 3500  # reference PC frequency (MHz)

EDGE_DEVICES = {
    "ESP32-S3": {
        "sram_kb": 512,
        "freq_mhz": 240,
        "description": "Espressif ESP32-S3, 512KB SRAM, 240MHz dual-core (+ 8MB PSRAM opt)"
    },
    "ARM_Cortex_M7": {
        "sram_kb": 1024,
        "freq_mhz": 480,
        "description": "STM32H7, 1MB SRAM, 480MHz"
    },
    "Raspberry_Pi_4B": {
        "sram_kb": 4 * 1024 * 1024,  # 4GB
        "freq_mhz": 1500,
        "description": "Raspberry Pi 4B, 4GB RAM, 1.5GHz quad-core ARM Cortex-A72"
    }
}


def analyze_edge_feasibility(model_results):
    """Analyze deployment feasibility of each model on edge devices."""
    feasibility = {}

    for device_name, device_info in EDGE_DEVICES.items():
        device_result = {
            "sram_kb": device_info["sram_kb"],
            "freq_mhz": device_info["freq_mhz"],
            "description": device_info["description"],
            "models": {}
        }

        for model_name, mdata in model_results.items():
            int8_kb = mdata["size_int8_kb"]
            # Runtime memory also needs activation + buffer, estimated as 3x INT8 size
            runtime_kb = int8_kb * 3
            fits = runtime_kb <= device_info["sram_kb"]

            # Inference latency estimate (scaled by frequency ratio)
            pc_latency_ms = mdata["inference_ms"]
            scale = REF_FREQ_MHZ / device_info["freq_mhz"]
            est_latency_ms = round(pc_latency_ms * scale, 3)

            device_result["models"][model_name] = {
                "fits": fits,
                "model_int8_kb": round(int8_kb, 2),
                "runtime_est_kb": round(runtime_kb, 2),
                "sram_available_kb": device_info["sram_kb"],
                "est_latency_ms": est_latency_ms,
                "feasible": fits
            }

        feasibility[device_name] = device_result

    return feasibility


# ========== Print Summary Table ==========

def print_summary_table(results):
    """Print a clear summary table."""
    models = list(results["parameter_counts"].keys())

    print("\n" + "=" * 100)
    print("  CBM-MCI Edge Computing Analysis Summary")
    print("=" * 100)

    # Parameter count & model size table
    print(f"\n{'Model':<22} {'Params':>10} {'FP32(KB)':>14} {'INT8(KB)':>14} {'Memory(KB)':>14} {'Inference(ms)':>14}")
    print("-" * 90)
    for m in models:
        params = results["parameter_counts"][m]
        fp32 = results["model_size_fp32_kb"][m]
        int8 = results["model_size_int8_kb"][m]
        mem = results["memory_footprint_kb"][m]
        inf = results["inference_time_cpu_ms"][m]
        print(f"  {m:<20} {params:>10,} {fp32:>14.2f} {int8:>14.2f} {mem:>14.2f} {inf:>14.4f}")

    # Edge device feasibility table
    print(f"\n{'Edge Device Feasibility'}:")
    print("-" * 90)
    print(f"{'Model':<22}", end="")
    for device in EDGE_DEVICES:
        print(f"  {device:<20}", end="")
    print()
    print(f"{'':22}", end="")
    for device_name, dinfo in EDGE_DEVICES.items():
        print(f"  {'('+str(dinfo['sram_kb'])+'KB SRAM)':<20}", end="")
    print()
    print("-" * 90)

    for m in models:
        print(f"  {m:<20}", end="")
        for device in EDGE_DEVICES:
            mdata = results["edge_feasibility"][device]["models"][m]
            status = "OK" if mdata["fits"] else "OOM"
            latency = f"{mdata['est_latency_ms']}ms"
            cell = f"{status}({latency})"
            print(f"  {cell:<20}", end="")
        print()

    print("=" * 100)
    print(f"\nNote: INT8 size = FP32 size / 4; Inference latency estimated by frequency scaling (ref: {REF_FREQ_MHZ}MHz)")
    print(f"    Edge deployment criterion: INT8 size x 3 (incl. activation buffer) <= device SRAM")
    print()


# ========== Main Flow ==========

def main():
    print("=" * 70)
    print("  CBM-MCI Edge Computing Analysis")
    print("=" * 70)

    # 1. Load data
    print("\n[Step 1] Loading data...")
    X, y = load_data()
    x_sample = X[0].astype(np.float32)
    n_features = X.shape[1]
    print(f"  Input dim: {n_features}, samples: {X.shape[0]}")

    # 2. Build models
    print("\n[Step 2] Building models...")

    # CBM models (using new instances; param count is independent of training)
    cbm_supervised = CBM_Supervised(n_demo=N_DEMO, n_classes=N_CLASSES)
    cbm_hybrid = CBM_Hybrid(n_demo=N_DEMO, n_classes=N_CLASSES, n_free_dims=N_FREE_DIMS)
    cbm_supervised.eval()
    cbm_hybrid.eval()

    # Baseline models (need fit for parameter counting)
    print("  Training baseline models...")
    lr = LogisticRegression(max_iter=200, C=1.0, random_state=42)
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)

    lr.fit(X, y)
    print("    LR fitted")
    svm.fit(X, y)
    print("    SVM fitted")
    rf.fit(X, y)
    print("    RF fitted")
    mlp.fit(X, y)
    print("    MLP fitted")

    if HAS_XGB:
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        xgb_model.fit(X, y)
        print("    XGBoost fitted")

    # 3. Count parameters
    print("\n[Step 3] Counting parameters...")
    param_counts = {
        "CBM-Supervised": count_torch_params(cbm_supervised),
        "CBM-Hybrid": count_torch_params(cbm_hybrid),
        "LR": count_lr_params(lr),
        "SVM": count_svm_params(svm),
        "RF": count_rf_params(rf),
        "MLP": count_mlp_params(mlp),
    }
    if HAS_XGB:
        param_counts["XGBoost"] = count_xgb_params(xgb_model)

    for k, v in param_counts.items():
        print(f"  {k:<22}: {v:,}")

    # 4. Model size
    print("\n[Step 4] Computing model sizes...")
    size_fp32 = {
        "CBM-Supervised": get_torch_model_size_kb(cbm_supervised),
        "CBM-Hybrid": get_torch_model_size_kb(cbm_hybrid),
        "LR": get_sklearn_model_size_kb(lr),
        "SVM": get_sklearn_model_size_kb(svm),
        "RF": get_sklearn_model_size_kb(rf),
        "MLP": get_sklearn_model_size_kb(mlp),
    }
    if HAS_XGB:
        size_fp32["XGBoost"] = get_sklearn_model_size_kb(xgb_model)

    size_int8 = {k: round(v / 4, 2) for k, v in size_fp32.items()}

    # 5. Inference latency
    print("\n[Step 5] Measuring inference latency (1000 repeats average)...")
    inference_ms = {}

    inference_ms["CBM-Supervised"] = measure_torch_inference_ms(cbm_supervised, x_sample)
    print(f"  CBM-Supervised: {inference_ms['CBM-Supervised']} ms")

    inference_ms["CBM-Hybrid"] = measure_torch_inference_ms(cbm_hybrid, x_sample)
    print(f"  CBM-Hybrid:     {inference_ms['CBM-Hybrid']} ms")

    for name, model in [("LR", lr), ("SVM", svm), ("RF", rf), ("MLP", mlp)]:
        inference_ms[name] = measure_sklearn_inference_ms(model, x_sample)
        print(f"  {name:<22}: {inference_ms[name]} ms")

    if HAS_XGB:
        inference_ms["XGBoost"] = measure_sklearn_inference_ms(xgb_model, x_sample)
        print(f"  XGBoost:        {inference_ms['XGBoost']} ms")

    # 6. Memory footprint
    print("\n[Step 6] Estimating memory footprint...")
    memory_kb = {
        "CBM-Supervised": estimate_torch_memory_kb(cbm_supervised, x_sample),
        "CBM-Hybrid": estimate_torch_memory_kb(cbm_hybrid, x_sample),
        "LR": estimate_sklearn_memory_kb(lr),
        "SVM": estimate_sklearn_memory_kb(svm),
        "RF": estimate_sklearn_memory_kb(rf),
        "MLP": estimate_sklearn_memory_kb(mlp),
    }
    if HAS_XGB:
        memory_kb["XGBoost"] = estimate_sklearn_memory_kb(xgb_model)

    # 7. Edge feasibility analysis
    print("\n[Step 7] Analyzing edge device deployment feasibility...")
    model_results = {}
    for m in param_counts.keys():
        model_results[m] = {
            "size_int8_kb": size_int8[m],
            "inference_ms": inference_ms[m],
        }
    edge_feasibility = analyze_edge_feasibility(model_results)

    # 8. Aggregate results
    results = {
        "parameter_counts": param_counts,
        "model_size_fp32_kb": size_fp32,
        "model_size_int8_kb": size_int8,
        "inference_time_cpu_ms": inference_ms,
        "memory_footprint_kb": memory_kb,
        "edge_feasibility": edge_feasibility,
        "meta": {
            "n_input_features": int(n_features),
            "n_concepts": N_CONCEPTS,
            "n_classes": N_CLASSES,
            "n_hidden_diag_head": N_HIDDEN,
            "n_free_dims_hybrid": N_FREE_DIMS,
            "ref_cpu_freq_mhz": REF_FREQ_MHZ,
            "inference_repeats": 1000
        }
    }

    # 9. Save JSON
    out_path = os.path.join(SCRIPT_DIR, 'results', 'edge_computing_analysis.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[Results] Saved to: {out_path}")

    # 10. Print summary table
    print_summary_table(results)

    # 11. Key findings report
    print("\n[Key Findings]")
    cbm_s_params = param_counts["CBM-Supervised"]
    cbm_h_params = param_counts["CBM-Hybrid"]
    cbm_s_int8 = size_int8["CBM-Supervised"]
    cbm_h_int8 = size_int8["CBM-Hybrid"]

    print(f"  CBM-Supervised params: {cbm_s_params:,}, INT8 size: {cbm_s_int8:.2f} KB")
    print(f"  CBM-Hybrid     params: {cbm_h_params:,}, INT8 size: {cbm_h_int8:.2f} KB")

    # Compare with RF/XGB
    rf_params = param_counts.get("RF", 0)
    print(f"  RF             params: {rf_params:,} equivalent params")

    # ESP32-S3 feasibility
    esp32_cbm_s = edge_feasibility["ESP32-S3"]["models"]["CBM-Supervised"]
    print(f"\n  ESP32-S3 (512KB SRAM):")
    print(f"    CBM-Supervised: {'Deployable' if esp32_cbm_s['fits'] else 'OOM'}, "
          f"runtime est. {esp32_cbm_s['runtime_est_kb']:.1f}KB, "
          f"inference latency ~{esp32_cbm_s['est_latency_ms']}ms")

    rpi_cbm_s = edge_feasibility["Raspberry_Pi_4B"]["models"]["CBM-Supervised"]
    print(f"\n  Raspberry Pi 4B (4GB RAM):")
    print(f"    CBM-Supervised: {'Deployable' if rpi_cbm_s['fits'] else 'OOM'}, "
          f"inference latency ~{rpi_cbm_s['est_latency_ms']}ms")

    return results


if __name__ == "__main__":
    main()
