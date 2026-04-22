import os
import sys
import multiprocessing
import pandas as pd
from qiskit import QuantumCircuit, qasm2

# ==============================
# CONFIG
# ==============================
QASM_FOLDER = sys.argv[1] if len(sys.argv) > 1 else "datav1/quantum_circuits"

# Backend mode: "aer" (CPU) | "aer_gpu" (GPU — requires qiskit-aer-gpu)
BACKEND_MODE = sys.argv[2] if len(sys.argv) > 2 else "aer"

SHOTS = 1024
TIMEOUT = 300  # seconds per circuit


# ==============================
# SUBPROCESS WORKER
# ==============================

def _subprocess_aer(qasm_str, shots, result_queue, use_gpu=False):
    """Aer simulation on CPU or GPU. No noise, no T1/T2."""
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        simulator = AerSimulator(device="GPU") if use_gpu else AerSimulator()
        qc = QuantumCircuit.from_qasm_str(qasm_str)
        if len(qc.clbits) == 0:
            qc.measure_all()
        job = simulator.run(qc, shots=shots)
        result = job.result()
        if not result.success:
            result_queue.put(False)
            return
        time_taken = result.results[0].time_taken
        result_queue.put(time_taken if time_taken > 0.0 else False)
    except Exception as e:
        result_queue.put(False)


# ==============================
# RUN ONE PASS
# ==============================

def run_all_circuits(run_name):
    if BACKEND_MODE not in ("aer", "aer_gpu"):
        print(f"Unknown backend mode: {BACKEND_MODE}. Use: aer | aer_gpu")
        sys.exit(1)

    use_gpu = BACKEND_MODE == "aer_gpu"
    out_dir = os.path.dirname(QASM_FOLDER.rstrip("/")) or "."
    prefix = BACKEND_MODE if use_gpu else os.path.basename(QASM_FOLDER.rstrip("/"))
    out_path = os.path.join(out_dir, f"{prefix}_{run_name}.csv")

    rows = []
    for file in sorted(os.listdir(QASM_FOLDER)):
        if not file.endswith(".qasm"):
            continue
        file_path = os.path.join(QASM_FOLDER, file)
        print(f"[{run_name}] {file}", flush=True)
        try:
            qc = QuantumCircuit.from_qasm_file(file_path)
            qasm_str = qasm2.dumps(qc)
        except Exception as e:
            print(f"   Skipped (load): {e}", flush=True)
            continue

        result_queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=_subprocess_aer,
                                    args=(qasm_str, SHOTS, result_queue, use_gpu))
        p.start()
        p.join(TIMEOUT)
        if p.is_alive():
            p.terminate(); p.join(5)
            if p.is_alive(): p.kill()
            print(f"   Skipped (timeout)", flush=True)
            continue

        time_taken = result_queue.get() if not result_queue.empty() else False
        if not time_taken:
            print(f"   Skipped (error/zero)", flush=True)
            continue

        rows.append({"circuit_name": os.path.splitext(file)[0],
                     "quantum_circuit": qasm_str,
                     "time_taken": time_taken,
                     "device": BACKEND_MODE})
        print(f"   time_taken: {time_taken:.6f}s", flush=True)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved → {out_path}\n", flush=True)


# ==============================
# COMBINE 3 RUNS
# ==============================

def combine_results():
    use_gpu = BACKEND_MODE == "aer_gpu"
    out_dir = os.path.dirname(QASM_FOLDER.rstrip("/")) or "."
    prefix = BACKEND_MODE if use_gpu else os.path.basename(QASM_FOLDER.rstrip("/"))

    df1 = pd.read_csv(os.path.join(out_dir, f"{prefix}_first_run.csv"))
    df2 = pd.read_csv(os.path.join(out_dir, f"{prefix}_second_run.csv"))
    df3 = pd.read_csv(os.path.join(out_dir, f"{prefix}_third_run.csv"))

    df1 = df1.rename(columns={"time_taken": "run1"})
    df2 = df2.rename(columns={"time_taken": "run2"})
    df3 = df3.rename(columns={"time_taken": "run3"})

    merged = df1.merge(df2[["circuit_name", "run2"]], on="circuit_name")
    merged = merged.merge(df3[["circuit_name", "run3"]], on="circuit_name")
    merged["time_taken"] = (merged["run1"] + merged["run2"] + merged["run3"]) / 3
    if "device" in df1.columns:
        merged["device"] = df1["device"]

    out_path = os.path.join(out_dir, f"{prefix}_final.csv")
    merged.to_csv(out_path, index=False, float_format="%.10f")
    print(f"Saved → {out_path}")


# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    print(f"Mode: {BACKEND_MODE} | Folder: {QASM_FOLDER}", flush=True)

    run_all_circuits("first_run")
    run_all_circuits("second_run")
    run_all_circuits("third_run")
    combine_results()
