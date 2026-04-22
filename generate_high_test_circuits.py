#!/usr/bin/env python3
"""
Generate new HIGH tier circuits (21–32q), benchmark on AerSimulator,
and append to test_100_final.csv, test_200_final.csv, test_300_final.csv.

Each test set gets unique circuits (different seeds).
  test_100 → seed offset 3000  (~2 circuits per qubit size)
  test_200 → seed offset 4000  (~2 circuits per qubit size)
  test_300 → seed offset 5000  (~2 circuits per qubit size)

Circuit types: RealAmplitudes / EfficientSU2 / TwoLocal (reps=1 for 21-32q)
Note: 33q+ exceed AerSimulator memory — max is 32q.
"""
import os, sys, random
import numpy as np
import pandas as pd
from pathlib import Path
import io
import qiskit.qasm2 as qasm2
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes, EfficientSU2, TwoLocal
from qiskit_aer import AerSimulator

sys.path.append(os.getcwd() + '/model')
sys.path.append(os.getcwd() + '/data_preparation')

random.seed(42)
np.random.seed(42)

SHOTS    = 1024
PASSES   = 3
BASIS_GATES = [
    'cx','u3','u2','u1','id','x','y','z','h','s','sdg','t','tdg',
    'rx','ry','rz','sx','sxdg','cz','cy','swap','ch','ccx','cswap',
    'crx','cry','crz','cu1','cp','cu3','csx','cu','rxx','rzz','rccx',
    'xx_plus_yy','ecr',
]

simulator = AerSimulator()
DATA_DIR  = Path("datav1")

# (tag, csv_file, seed_offset, n_per_qubit)
TEST_SETS = [
    ("test_100", DATA_DIR / "test_100_final.csv", 3000, 2),
    ("test_200", DATA_DIR / "test_200_final.csv", 4000, 2),
    ("test_300", DATA_DIR / "test_300_final.csv", 5000, 2),
]

# Cycle through 3 circuit types
CIRCUIT_TYPES = ["realamp", "su2", "twolocal"]

HIGH_QUBITS = list(range(21, 33))   # 21 to 32 inclusive


def make_circuit(n: int, ctype: str, seed: int) -> QuantumCircuit:
    np.random.seed(seed)
    if ctype == "realamp":
        qc_t = RealAmplitudes(n, reps=1)
    elif ctype == "su2":
        qc_t = EfficientSU2(n, reps=1)
    else:
        qc_t = TwoLocal(n, ['ry', 'rz'], 'cx', reps=1)
    params = qc_t.parameters
    values = {p: np.random.uniform(0, 2 * np.pi) for p in params}
    qc = qc_t.assign_parameters(values)
    qc = transpile(qc, basis_gates=BASIS_GATES, optimization_level=0)
    return qc


def to_qasm_str(qc: QuantumCircuit) -> str:
    buf = io.StringIO()
    qasm2.dump(qc, buf)
    return buf.getvalue()


def run_aer(qc: QuantumCircuit):
    """Run PASSES times; returns list of floats (may be empty on OOM)."""
    qc_run = qc.copy()
    if len(qc_run.clbits) == 0:
        qc_run.measure_all()
    times = []
    for _ in range(PASSES):
        try:
            result = simulator.run(qc_run, shots=SHOTS).result()
            t = result.results[0].time_taken
            if t > 0:
                times.append(t)
        except Exception as e:
            print(f"      Aer error: {e}", flush=True)
            break
    return times


# ── Main generation loop ───────────────────────────────────────────────────────
for tag, csv_path, seed_offset, n_per_q in TEST_SETS:
    print(f"\n{'='*60}")
    print(f"  {tag.upper()}  — adding HIGH circuits (21–32q)")
    print(f"{'='*60}")

    df_existing    = pd.read_csv(csv_path)
    existing_names = set(df_existing["circuit_name"].tolist())
    new_rows       = []
    ct_cycle       = 0

    for n in HIGH_QUBITS:
        for k in range(n_per_q):
            ctype = CIRCUIT_TYPES[ct_cycle % len(CIRCUIT_TYPES)]
            ct_cycle += 1
            seed  = seed_offset + n * 10 + k
            name  = f"high_indep_{ctype}_{n}_{tag}_s{seed}"

            if name in existing_names:
                print(f"  SKIP  {name}", flush=True)
                continue

            print(f"  {name:<52}", end=" ", flush=True)
            try:
                qc = make_circuit(n, ctype, seed)
            except Exception as e:
                print(f"BUILD ERR: {e}", flush=True)
                continue

            qs    = to_qasm_str(qc)
            times = run_aer(qc)

            if len(times) == PASSES:
                avg_t = float(np.mean(times))
                print(f"{avg_t*1000:>10.2f} ms  ✓", flush=True)
                new_rows.append({
                    "circuit_name":    name,
                    "quantum_circuit": qs,
                    "run1":            times[0],
                    "run2":            times[1],
                    "run3":            times[2],
                    "time_taken":      avg_t,
                })
            elif len(times) > 0:
                avg_t = float(np.mean(times))
                print(f"{avg_t*1000:>10.2f} ms  (partial {len(times)}/{PASSES})", flush=True)
                new_rows.append({
                    "circuit_name":    name,
                    "quantum_circuit": qs,
                    "run1":            times[0] if len(times) > 0 else None,
                    "run2":            times[1] if len(times) > 1 else None,
                    "run3":            times[2] if len(times) > 2 else None,
                    "time_taken":      avg_t,
                })
            else:
                print(f"{'OOM/failed':>10}       — skipped", flush=True)

    if new_rows:
        df_out = pd.concat([df_existing, pd.DataFrame(new_rows)], ignore_index=True)
        df_out.to_csv(csv_path, index=False)
        print(f"\n  Added {len(new_rows)} circuits → {csv_path}  (total {len(df_out)})", flush=True)
    else:
        print(f"\n  Nothing added to {tag}.", flush=True)

print("\nDone — HIGH circuit generation complete.")
