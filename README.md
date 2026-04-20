# CBM-MoCA: Concept Bottleneck Models for MoCA-Based Cognitive Impairment Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Overview

CBM-MoCA is an interpretable deep learning framework built upon **Concept Bottleneck Models (CBMs)** for three-class cognitive impairment classification using the **Montreal Cognitive Assessment (MoCA)** scale. The framework maps raw MoCA sub-item scores into seven clinically-defined cognitive domains before producing a final diagnostic prediction, providing per-concept interpretability directly aligned with neuropsychological practice.

Two model variants are provided:

| Variant | Description |
|---|---|
| **CBM-Supervised** | Strictly interpretable; all reasoning is routed through the concept bottleneck layer with no residual information leakage. |
| **CBM-Hybrid** | Balances predictive performance and interpretability by allowing a small residual pathway alongside the concept layer. |

Both variants are validated on two independent cohorts and support three diagnostic categories:

- **Normal Cognition (NC)**
- **Mild Cognitive Impairment (MCI)**
- **Dementia**

---

## Key Features

- **MoCA-aligned concept bottleneck layer** — seven cognitive domain concepts derived directly from MoCA subscores (Visuospatial/Executive, Naming, Attention, Language, Abstraction, Delayed Recall, Orientation)
- **Dual-dataset validation** — NACC (Alzheimer's disease cohort) and PPMI (Parkinson's disease cohort)
- **Three-class classification** — Normal Cognition / MCI / Dementia
- **Fairness analysis** — subgroup evaluation stratified by education level
- **Edge deployment feasibility** — INT8 quantization analysis targeting ESP32-S3 microcontroller

---

## Key Results

| Model | Dataset | Balanced Accuracy |
|---|---|---|
| CBM-Supervised | NACC | 97.80% |
| CBM-Supervised | PPMI | 85.49% |
| CBM-Hybrid | NACC | **99.38%** |
| CBM-Hybrid | PPMI | **88.27%** |

---

## Repository Structure

```
CBM-Moca/
├── data/                        # Data directory (not included; see data/README.md)
│   └── README.md                # Data acquisition instructions
├── results/                     # Experiment outputs (JSON/CSV)
├── model_dcm_dnn.py             # Core CBM model definitions (CBM-Supervised, CBM-Hybrid)
├── experiment_data.py           # Data loading and preprocessing utilities
├── run_experiment.py            # Main experiment runner (Exp 1: BCR/BA metrics)
├── run_cbm_sota.py              # Experiment 2: comparison with SOTA baselines
├── run_cbm_sensitivity.py       # Experiment 3: concept sensitivity analysis
├── run_cbm_stratified.py        # Experiment 4: stratified (fairness) evaluation
├── run_cbm_vs_shap.py           # Experiment 5: CBM concepts vs. SHAP explanations
├── run_cbm_case_study.py        # Experiment 6: individual prediction case study
├── run_cbm_clinical_alignment.py# Experiment 7: clinical concept alignment
├── run_edge_computing_analysis.py# Experiment 8: edge deployment feasibility
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md                    # This file
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- See `requirements.txt` for the full dependency list

---

## Installation

```bash
git clone https://github.com/<your-username>/CBM-Moca.git
cd CBM-Moca
pip install -r requirements.txt
```

---

## Data Acquisition

This repository does **not** include raw data files. Please refer to [`data/README.md`](data/README.md) for detailed instructions on how to apply for and preprocess the NACC and PPMI datasets.

---

## Reproducing Experiments

All experiments use 10-fold stratified cross-validation. Run each script from the repository root:

| Script | Description | Paper Section |
|---|---|---|
| `python run_experiment.py` | Exp 1 – Main classification results (BCR / Balanced Accuracy) | §4.1 |
| `python run_cbm_sota.py` | Exp 2 – Comparison with SOTA baselines (PlainDNN, TabNet, XGBoost, etc.) | §4.2 |
| `python run_cbm_sensitivity.py` | Exp 3 – Concept sensitivity / ablation analysis | §4.3 |
| `python run_cbm_stratified.py` | Exp 4 – Fairness evaluation (education-level subgroups) | §4.4 |
| `python run_cbm_vs_shap.py` | Exp 5 – CBM concept weights vs. SHAP feature importance | §4.5 |
| `python run_cbm_case_study.py` | Exp 6 – Individual-level counterfactual case study | §4.6 |
| `python run_cbm_clinical_alignment.py` | Exp 7 – Clinical concept polarity alignment | §4.7 |
| `python run_edge_computing_analysis.py` | Exp 8 – Edge deployment latency estimation (ESP32-S3, INT8) | §4.8 |

Results will be saved to the `results/` directory in JSON/CSV format.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{cbm_moca_2026,
  title   = {Concept Bottleneck Models for MoCA-Based Cognitive Impairment Classification},
  author  = {[Author Name]},
  journal = {Neurocomputing},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
