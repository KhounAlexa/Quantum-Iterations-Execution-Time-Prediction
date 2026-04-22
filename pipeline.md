# QET Prediction — Pipeline Guide

Run the full pipeline from scratch: collect → build dataset → train → evaluate.  
Everything is isolated under `runs/<name>/` — nothing touches existing data.

---

## Installation

### Option A — Conda (recommended)
```bash
conda env create -f environment.yml
conda activate qet-env
```

### Option B — pip
```bash
pip install -r requirements.txt
```

### For GPU support
GPU accelerates **model training** significantly and can also run AerSimulator on GPU.  
Requires a CUDA-capable GPU (NVIDIA).

```bash
# Replace qiskit-aer with GPU build
pip uninstall qiskit-aer
pip install qiskit-aer-gpu

# Install CUDA-compatible PyTorch (example: CUDA 11.8)
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.7.0
```

> The pipeline detects CUDA automatically for training.  
> For GPU-accelerated Aer simulation, use `--mode aer_gpu`.

### Key dependencies
| Package | Version | Purpose |
|---|---|---|
| qiskit | 2.3.1 | Quantum circuit framework |
| qiskit-aer | 0.17.2 | AerSimulator (CPU) |
| qiskit-aer-gpu | 0.17.2 | AerSimulator (GPU) — optional |
| qiskit-ibm-runtime | latest | Fake backends (optional) |
| torch | 2.11.0 | Model training |
| torch-geometric | 2.7.0 | Graph neural network |
| torchpack | 0.3.1 | Config / training utilities |
| numpy | 2.4.4 | Numerical computing |
| pandas | 3.0.1 | Data handling |
| scikit-learn | 1.6.1 | Evaluation metrics |
| networkx | latest | Circuit DAG conversion |
| rustworkx | latest | Circuit DAG conversion |

---

## Usage

```bash
python pipeline.py --name <run_name> --mode <aer|aer_gpu> [options]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--name` | required | Your run name, e.g. `john_cpu` |
| `--mode` | `aer` | `aer` = CPU, `aer_gpu` = GPU |
| `--timeout` | `300` | Max seconds per circuit before skip |
| `--skip-collect` | off | Skip data collection (CSV already exists) |
| `--skip-train` | off | Skip training (model already exists) |
| `--skip-eval` | off | Skip evaluation |

### Examples

```bash
# Full run — CPU
python pipeline.py --name my_run --mode aer

# Full run — GPU
python pipeline.py --name my_run --mode aer_gpu

# Resume after crash (collection already done)
python pipeline.py --name my_run --mode aer --skip-collect

# Skip collection and training — evaluate only
python pipeline.py --name my_run --mode aer --skip-collect --skip-train

# Custom timeout (5 min per circuit)
python pipeline.py --name my_run --mode aer --timeout 300
```

---

## What it does — step by step

### Step 1 — Collect
Runs every QASM circuit in `datav1/quantum_circuits/` through AerSimulator **3 times**, then averages.

- Saves `runs/<name>/<name>_first_run.csv`, `_second_run.csv`, `_third_run.csv`
- Saves final averaged `runs/<name>/<name>_final.csv`
- Circuits that timeout or error are skipped automatically
- **Resumable**: if a trial CSV already exists it is skipped

### Step 2 — Build dataset
Converts circuits to graph features (DAG representation), standardizes, saves.

- Saves `runs/<name>/<name>_training_data.npy`
- Saves `runs/<name>/<name>_stats.pth`

### Step 3 — Train
Trains a TransformerConv GNN from scratch using the built dataset.

- Config: `model/parameter/default/config.yaml`
- Saves `runs/<name>/<name>_model.pth`
- Uses GPU automatically if available

### Step 4 — Evaluate
Runs your model against all 5 test sets: `test_50`, `test_100`, `test_200`, `test_300`, `demo_mixed`.

- Saves results to `runs/<name>/results/`
- Metrics: R², MAE, within-15% accuracy per test set

---

## Output structure

```
runs/
  <name>/
    <name>_first_run.csv       ← trial 1 raw timings
    <name>_second_run.csv      ← trial 2 raw timings
    <name>_third_run.csv       ← trial 3 raw timings
    <name>_final.csv           ← averaged timings (used for training)
    <name>_training_data.npy   ← graph dataset
    <name>_stats.pth           ← feature normalization stats
    <name>_model.pth           ← trained model weights
    results/
      eval_<name>_test_50.csv / .txt
      eval_<name>_test_100.csv / .txt
      eval_<name>_test_200.csv / .txt
      eval_<name>_test_300.csv / .txt
      eval_<name>_demo_mixed.csv / .txt
```

---

## Hardware requirements

| Use case | RAM | CPU | GPU |
|---|---|---|---|
| Collect + train (≤20q) | 8 GB | 4 cores | Optional |
| Collect + train (≤32q) | 72 GB | 8+ cores | Optional |
| GPU-accelerated training | 8 GB | Any | NVIDIA CUDA |
| GPU-accelerated Aer (≤28q) | 8 GB | Any | 8+ GB VRAM |

> RAM is the bottleneck for large-qubit simulation, not GPU.  
> 32q statevector simulation requires ~64 GB RAM.
