# Quantum Execution Time (QET) Prediction

A graph transformer model that predicts quantum circuit execution time on AerSimulator, achieving **R² > 0.99** on HIGH-qubit circuits (15–32 qubits) and **R² > 0.93** overall.

---

## Hardware Requirements

### Our Development Machine
| Component | Spec |
|---|---|
| RAM | 72 GB |
| CPU | 12 cores |
| GPU | None (CPU-only training & inference) |
| OS | Ubuntu Linux |

### What You Need — By Use Case

| Use Case | Min RAM | Min CPU | GPU |
|---|---|---|---|
| **Predict only** (run model on your circuits) | 4 GB | Any | Not needed |
| **Run eval_all.py** on test sets (≤20q Aer) | 8 GB | 4 cores | Optional |
| **Run eval_all.py** with demo_mixed (≤30q Aer) | 16 GB | 4+ cores | Optional |
| **Aer simulation up to 32q** | **72 GB** | 8+ cores | Not practical (see note) |
| **Re-train V7 model** | 8 GB | 4+ cores | Strongly recommended |

> **RAM is the critical resource, not GPU.** AerSimulator uses statevector simulation internally which requires RAM proportional to 2^n:
> - 20q → ~16 MB RAM
> - 25q → ~512 MB RAM
> - 28q → ~4 GB RAM
> - 30q → ~16 GB RAM
> - 32q → ~64 GB RAM ← this is our limit

> **GPU for Aer simulation (31–32q):** Even with `qiskit-aer-gpu`, you need VRAM equivalent to the RAM requirement above. No consumer GPU has 64 GB VRAM, so 32q simulation must run on CPU RAM. GPU-accelerated Aer is useful up to ~28q.

### If Your Machine Has Less RAM

| Your RAM | What Works |
|---|---|
| 4–8 GB | Model inference + test sets ≤20q. Skip demo_mixed or use --skip-aer flag |
| 16–32 GB | Full eval up to 30q. 31–32q Aer will OOM |
| 32–64 GB | Full eval up to 31q |
| 72 GB+ | Full eval including 32q (our tested configuration) |

### GPU Support (Model Training & Inference)
The model runs on GPU automatically if CUDA is available. This speeds up **training significantly** (10–20×) but has minimal impact on inference since circuits are evaluated one-by-one via Aer anyway.

**CPU training time (our machine, 12 cores):** ~6–8 hours for 500 epochs  
**GPU training time (estimated, RTX 3090):** ~30–45 minutes

To enable GPU:
```bash
# Install GPU build of PyTorch (CUDA 11.8 example)
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.7.0

# For GPU-accelerated Aer simulation (up to ~28q)
pip install qiskit-aer-gpu   # replaces qiskit-aer
```
The code detects CUDA automatically — no code changes needed.

---

## Quick Start

### 1. Clone and set up environment
```bash
git clone <repo-url>
cd Quantum-Execution-Time-Prediction

# Option A — conda (recommended)
conda env create -f environment.yml
conda activate qet-env

# Option B — pip
pip install -r requirements.txt
```

### 2. Use the pre-trained V7 model to evaluate
```bash
python eval_all.py
```
This runs the V7 model on all test sets (test_50/100/200/300) and the demo_mixed circuits, then prints a V4 vs V6 vs V7 comparison. Results saved to `results_v7/full_result/`.

### 3. Train from scratch
```bash
cd model
python train.py default
```

---

## Repository Structure

```
├── model/
│   ├── train.py                  # Training entry point
│   ├── trainer.py                # Training loop
│   ├── builder.py                # Model / dataset / optimizer factory
│   ├── circs.py                  # Dataset loader (stratified split)
│   ├── transformer_model.py      # TransformerConv GNN architecture
│   └── parameter/default/
│       └── config.yaml           # Hyperparameters
│
├── data_preparation/             # Feature extraction pipeline
│   ├── execution.py              # Aer runner + graph feature builder
│   ├── circ_dag_converter.py     # QASM → DAG graph
│   └── helper.py                 # Standardisation utilities
│
├── datav1/                       # Processed data (not in git LFS)
│   ├── training_data_log.npy     # Pre-built graph dataset
│   ├── standardization_stats.pth # Feature mean/std
│   ├── split_indices.pth         # Train/val/test split (seed=1234)
│   ├── demo_mixed/               # 42 demo circuits (QASM files, 2–32q)
│   ├── test_50_final.csv         # External test set  (50 circuits)
│   ├── test_100_final.csv        # External test set (124 circuits, incl. 21–32q)
│   ├── test_200_final.csv        # External test set (209 circuits, incl. 21–32q)
│   └── test_300_final.csv        # External test set (309 circuits, incl. 21–32q)
│
├── results_v7/full_result/       # V7 model weights + evaluation results
│   ├── model.pth
│   ├── standardization_stats.pth
│   ├── aer_vs_v7_test_*.{csv,txt}
│   └── aer_vs_v7_demo_mixed_ext.{csv,txt}
│
├── eval_all.py                   # Unified evaluation script (all test sets + demo_mixed)
├── generate_high_test_circuits.py # Generate + benchmark new HIGH-qubit circuits
├── requirements.txt
└── environment.yml
```

---

## Model Architecture

```
Input: Quantum circuit DAG
  ├── Node features  : gate type, qubit index, gate index  (standardised)
  └── Global features: circuit-level statistics (43 features, standardised)

Graph Encoder: 4 × TransformerConv layers + global mean pooling
Global Branch : Linear projection of 43 global features
Fusion MLP    : 4-layer MLP combining graph + global features
Output        : log(execution_time_seconds)  →  exp(output) × 1000 = ms
```

**V7 config:** 4 layers · dropout=0.1 · Adam lr=3e-4 · cosine LR · 500 epochs · batch=32  
**Params:** 966,785

---

## Results (V7 Model)

| Test Set | Circuits | Overall R² | HIGH R² | ✓ within 15% |
|---|---|---|---|---|
| test_50  | 50  | 0.9539 | 0.9619 | 28/50  |
| test_100 | 124 | 0.9924 | 0.9929 | 76/124 |
| test_200 | 209 | 0.9896 | 0.9935 | 118/209 |
| test_300 | 309 | 0.9859 | 0.9936 | 184/309 |
| demo_mixed (2–30q) | 40 | 0.9955 | 0.9916 | 29/40 |

> **Note on LOW tier (2–6q):** Prediction accuracy is weaker for very small circuits due to dataset imbalance (small circuits run in <1ms, making relative error large). For production use on 2–6q circuits, consider V4 or V6 models.

---

## Version History

| Version | Layers | Dropout | Dataset | Notes |
|---|---|---|---|---|
| V4 | 3 | 0.0 | Smaller (≤20q) | Best on LOW/MED tier |
| V6 | 3 | 0.1 | Combined (≤30q) | Balanced across tiers |
| **V7** | **4** | **0.1** | **Combined (≤32q)** | **Best overall, current** |

---

## Data Source

Circuit benchmarks sourced from [MQTBench](https://www.cda.cit.tum.de/mqtbench/) (Chair of Quantum Technologies, TU Munich).

## Acknowledgements

- [mqt-bench](https://github.com/cda-tum/mqt-bench)
- [mqt-predictor](https://github.com/cda-tum/mqt-predictor)
- [torchquantum](https://github.com/mit-han-lab/torchquantum)
