#!/usr/bin/env python3
"""
QET Prediction — full pipeline from scratch.

Usage:
  python pipeline.py --name my_run --mode aer
  python pipeline.py --name my_run --mode aer_gpu   # requires qiskit-aer-gpu

Everything saved to:
  runs/<name>/
    <name>_first_run.csv
    <name>_second_run.csv
    <name>_third_run.csv
    <name>_final.csv
    <name>_training_data.npy
    <name>_stats.pth
    <name>_model.pth
    results/
      eval_<name>_<test_set>.{csv,txt}

Each step skips automatically if its output already exists — safe to resume.
"""

import argparse, os, sys, multiprocessing, pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, os.getcwd() + "/data_preparation")
sys.path.insert(0, os.getcwd() + "/model")
sys.path.insert(0, os.getcwd())

CIRCUIT_DIR  = Path("datav1/quantum_circuits")
TEST_SETS    = [
    ("test_50",  Path("datav1/test_50_final.csv")),
    ("test_100", Path("datav1/test_100_final.csv")),
    ("test_200", Path("datav1/test_200_final.csv")),
    ("test_300", Path("datav1/test_300_final.csv")),
]
DEMO_DIR     = Path("datav1/demo_mixed")
SHOTS        = 1024
CLOSE_THRESH = 15.0


def hdr(n, title):
    print(f"\n{'='*60}\n  STEP {n}: {title}\n{'='*60}", flush=True)


# ============================================================
# STEP 1 — COLLECT
# ============================================================

def _worker(qasm_str, shots, q, use_gpu):
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        sim = AerSimulator(device="GPU") if use_gpu else AerSimulator()
        qc = QuantumCircuit.from_qasm_str(qasm_str)
        if len(qc.clbits) == 0:
            qc.measure_all()
        r = sim.run(qc, shots=shots).result()
        t = r.results[0].time_taken if r.success else 0.0
        q.put(t if t > 0.0 else False)
    except Exception:
        q.put(False)


def collect(run_dir, name, mode, timeout):
    hdr(1, f"Collect execution times  [{mode}]")
    final_csv = run_dir / f"{name}_final.csv"
    if final_csv.exists():
        print(f"  Already done: {final_csv}", flush=True)
        return final_csv

    use_gpu = mode == "aer_gpu"
    files = sorted(f for f in CIRCUIT_DIR.iterdir() if f.suffix == ".qasm")

    for trial in ("first_run", "second_run", "third_run"):
        out = run_dir / f"{name}_{trial}.csv"
        if out.exists():
            print(f"  {trial} already saved — skipping.", flush=True)
            continue

        rows = []
        for file in files:
            print(f"  [{trial}] {file.name}", flush=True)
            try:
                from qiskit import QuantumCircuit, qasm2
                qc = QuantumCircuit.from_qasm_file(str(file))
                qasm_str = qasm2.dumps(qc)
            except Exception as e:
                print(f"    Skipped (load): {e}", flush=True)
                continue

            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=_worker, args=(qasm_str, SHOTS, q, use_gpu))
            p.start(); p.join(timeout)
            if p.is_alive():
                p.terminate(); p.join(5)
                if p.is_alive(): p.kill()
                print(f"    Skipped (timeout {timeout}s)", flush=True)
                continue

            t = q.get() if not q.empty() else False
            if not t:
                print(f"    Skipped (error/zero)", flush=True)
                continue

            rows.append({"circuit_name": file.stem, "quantum_circuit": qasm_str,
                         "time_taken": t, "device": mode})
            print(f"    {t:.6f}s", flush=True)

        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  Saved {len(rows)} circuits → {out}", flush=True)

    # Average 3 trials
    df1 = pd.read_csv(run_dir / f"{name}_first_run.csv").rename(columns={"time_taken": "run1"})
    df2 = pd.read_csv(run_dir / f"{name}_second_run.csv").rename(columns={"time_taken": "run2"})
    df3 = pd.read_csv(run_dir / f"{name}_third_run.csv").rename(columns={"time_taken": "run3"})
    merged = df1.merge(df2[["circuit_name","run2"]], on="circuit_name")
    merged = merged.merge(df3[["circuit_name","run3"]], on="circuit_name")
    merged["time_taken"] = (merged["run1"] + merged["run2"] + merged["run3"]) / 3
    merged["device"] = mode
    merged[["quantum_circuit","time_taken","device"]].to_csv(final_csv, index=False, float_format="%.10f")
    print(f"\n  Final CSV → {final_csv}  ({len(merged)} circuits)", flush=True)
    return final_csv


# ============================================================
# STEP 2 — BUILD DATASET
# ============================================================

def build_dataset(run_dir, name, csv_path):
    hdr(2, "Build graph dataset")
    npy_out = run_dir / f"{name}_training_data.npy"
    pth_out = run_dir / f"{name}_stats.pth"

    if npy_out.exists() and pth_out.exists():
        print(f"  Already done: {npy_out}", flush=True)
        return npy_out, pth_out

    from data_preparation import execution, helper

    df = pd.read_csv(csv_path)
    df = df[["quantum_circuit","time_taken"]].copy()
    df["device"] = "aer"
    print(f"  {len(df)} rows loaded", flush=True)

    print("  Extracting features ...", flush=True)
    sample, time_taken = execution.Execution().generate_training_sample_execution_time(df)
    print(f"  {len(sample)} samples extracted", flush=True)

    refined, non_zero_indices = helper.refine_training_data(sample)
    std_data, stats = helper.standardization_training_data(refined)

    print(f"  Node feature shape : {std_data[0].x.shape}", flush=True)
    print(f"  Global feature shape: {std_data[0].global_features.shape}", flush=True)

    for i in range(len(std_data)):
        std_data[i].y = float(np.log(float(time_taken[i]) + 1e-8))

    with open(npy_out, "wb") as f:
        pickle.dump(std_data, f)
    torch.save({"stats": stats, "non_zero_indices": non_zero_indices}, str(pth_out))

    print(f"  Saved: {npy_out}", flush=True)
    print(f"  Saved: {pth_out}", flush=True)
    return npy_out, pth_out


# ============================================================
# STEP 3 — TRAIN
# ============================================================

def train_model(run_dir, name, npy_path, pth_path):
    hdr(3, "Train model")
    model_out = run_dir / f"{name}_model.pth"
    if model_out.exists():
        print(f"  Already done: {model_out}", flush=True)
        return model_out

    import builder, trainer as tr
    from torchpack.utils.config import configs

    configs.load("model/parameter/default/config.yaml", recursive=True)
    configs.parameter_name = "default"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}", flush=True)

    with open(npy_path, "rb") as f:
        dataset = pickle.load(f)

    model   = builder.make_model().to(device)
    opt, sch = builder.make_optimizer(model), builder.make_scheduler(None)

    t = tr.Trainer(model=model, optimizer=opt, scheduler=sch,
                   dataset=dataset, device=device,
                   stats_path=str(pth_path), out_dir=str(run_dir), name=name)
    t.train()

    print(f"  Model saved → {model_out}", flush=True)
    return model_out


# ============================================================
# STEP 4 — EVALUATE
# ============================================================

def evaluate(run_dir, name, model_path, pth_path):
    hdr(4, "Evaluate model")
    from data_preparation import execution, helper
    import builder
    from torch_geometric.loader import DataLoader
    from torchpack.utils.config import configs

    res_dir = run_dir / "results"
    res_dir.mkdir(exist_ok=True)

    configs.load("model/parameter/default/config.yaml", recursive=True)
    configs.parameter_name = "default"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = builder.make_model().to(device)
    model.load_state_dict(torch.load(str(model_path), weights_only=True))
    model.eval()

    saved = torch.load(str(pth_path), weights_only=False)

    def get_nqubits(n):
        for p in n.split("_"):
            try: return int(p)
            except: pass
        return 0

    def get_tier(nq):
        if nq <= 6:  return "LOW"
        if nq <= 14: return "MED"
        if nq <= 30: return "HIGH"
        return "EXTRAP"

    def predict(df_feat):
        sample, times = execution.Execution().generate_training_sample_execution_time(df_feat)
        refined, _   = helper.refine_training_data(sample, non_zero_indices=saved["non_zero_indices"])
        std_data, _  = helper.standardization_training_data(refined, stats=saved["stats"])
        for i in range(len(std_data)):
            std_data[i].y = float(np.log(float(times[i]) + 1e-8))
            if std_data[i].global_features.dim() == 1:
                std_data[i].global_features = std_data[i].global_features.unsqueeze(0)
        loader = DataLoader(std_data, batch_size=1, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                preds.append(model(batch).item())
        return np.array(preds), np.array([float(t) for t in times])

    def print_metrics(tag, df_out, txt_lines):
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        y_true = df_out["aer_ms"].values
        y_pred = df_out["pred_ms"].values
        r2   = r2_score(np.log(y_true+1e-8), np.log(y_pred+1e-8))
        mse  = mean_squared_error(y_true, y_pred)
        mae  = mean_absolute_error(y_true, y_pred)
        within = (np.abs(y_true - y_pred) / (y_true + 1e-8) * 100 < CLOSE_THRESH).sum()
        line = f"{tag:12s}  R²={r2:.4f}  MAE={mae:.2f}ms  within15%={within}/{len(y_true)}"
        print(f"  {line}", flush=True)
        txt_lines.append(line)

    # --- Test sets ---
    for tag, csv_path in TEST_SETS:
        if not csv_path.exists():
            print(f"  Skipping {tag} (not found: {csv_path})", flush=True)
            continue
        print(f"\n  Evaluating {tag} ...", flush=True)
        df = pd.read_csv(csv_path)
        df["device"] = "aer"
        feat_df = df[["quantum_circuit","time_taken","device"]].copy()
        feat_df.columns = ["quantum_circuit","time_taken","device"]

        preds_log, times = predict(feat_df)
        pred_ms = np.exp(preds_log) * 1000
        aer_ms  = times * 1000

        rows = []
        for i, row in enumerate(df.itertuples()):
            nq   = get_nqubits(str(row.Index))
            rows.append({"circuit": i, "aer_ms": aer_ms[i], "pred_ms": pred_ms[i],
                         "tier": get_tier(nq),
                         "within_15": abs(aer_ms[i]-pred_ms[i])/(aer_ms[i]+1e-8)*100 < CLOSE_THRESH})

        out_df = pd.DataFrame(rows)
        out_df.to_csv(res_dir / f"eval_{name}_{tag}.csv", index=False)
        txt = []
        print_metrics(tag, out_df, txt)
        (res_dir / f"eval_{name}_{tag}.txt").write_text("\n".join(txt))

    # --- Demo mixed ---
    if DEMO_DIR.exists():
        print(f"\n  Evaluating demo_mixed ...", flush=True)
        from qiskit import QuantumCircuit, qasm2
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        rows = []
        for qf in sorted(DEMO_DIR.iterdir()):
            if qf.suffix != ".qasm": continue
            try:
                qc = QuantumCircuit.from_qasm_file(str(qf))
                tqc = qc.copy(); tqc.measure_all() if len(qc.clbits)==0 else None
                r = sim.run(tqc, shots=SHOTS).result()
                t = r.results[0].time_taken
                rows.append({"quantum_circuit": qasm2.dumps(qc), "time_taken": t, "device": "aer"})
            except Exception as e:
                print(f"    Skipped {qf.name}: {e}", flush=True)

        if rows:
            feat_df = pd.DataFrame(rows)
            preds_log, times = predict(feat_df)
            pred_ms = np.exp(preds_log) * 1000
            aer_ms  = times * 1000
            out_rows = [{"circuit": r["quantum_circuit"][:40], "aer_ms": aer_ms[i],
                         "pred_ms": pred_ms[i],
                         "within_15": abs(aer_ms[i]-pred_ms[i])/(aer_ms[i]+1e-8)*100 < CLOSE_THRESH}
                        for i, r in enumerate(rows)]
            out_df = pd.DataFrame(out_rows)
            out_df.to_csv(res_dir / f"eval_{name}_demo_mixed.csv", index=False)
            txt = []
            print_metrics("demo_mixed", out_df, txt)
            (res_dir / f"eval_{name}_demo_mixed.txt").write_text("\n".join(txt))

    print(f"\n  Results saved to: {res_dir}/", flush=True)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QET Prediction Pipeline")
    parser.add_argument("--name",    required=True, help="Run name, e.g. 'john_cpu'")
    parser.add_argument("--mode",    choices=["aer","aer_gpu"], default="aer")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per circuit (s)")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-train",   action="store_true")
    parser.add_argument("--skip-eval",    action="store_true")
    args = parser.parse_args()

    run_dir = Path("runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Name    : {args.name}")
    print(f"  Mode    : {args.mode}")
    print(f"  Output  : {run_dir}/")
    print(f"  Circuits: {CIRCUIT_DIR}/  ({sum(1 for f in CIRCUIT_DIR.iterdir() if f.suffix=='.qasm')} files)")
    print(f"  Timeout : {args.timeout}s per circuit")

    # Step 1
    csv_path = collect(run_dir, args.name, args.mode, args.timeout) if not args.skip_collect \
               else run_dir / f"{args.name}_final.csv"

    # Step 2
    npy_path, pth_path = build_dataset(run_dir, args.name, csv_path)

    # Step 3
    model_path = train_model(run_dir, args.name, npy_path, pth_path) if not args.skip_train \
                 else run_dir / f"{args.name}_model.pth"

    # Step 4
    if not args.skip_eval:
        evaluate(run_dir, args.name, model_path, pth_path)

    print(f"\n{'='*60}")
    print(f"  Done! Everything in: {run_dir}/")
    print(f"{'='*60}\n")
