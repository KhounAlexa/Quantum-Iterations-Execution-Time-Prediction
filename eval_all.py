#!/usr/bin/env python3
"""
eval_all.py — single unified evaluation script for the QET Prediction project.

Evaluates V7 model (4 layers, dropout=0.1) on:
  1. External test sets  : test_50 / test_100 / test_200 / test_300
  2. Demo-mixed circuits : 2–30q (in-distribution) + 31–32q (EXTRAP)
                           AerSimulator is re-run each time for fresh timings.

For every test:
  • Tier-grouped table  (LOW ≤6q / MED 7–14q / HIGH 15+q / EXTRAP 31–32q)
  • ✓/✗ per circuit     (threshold: 15 % relative error)
  • Full metrics        R², MSE, NMSE, MAE, Max AE

Results saved to:
  results_v7/full_result/aer_vs_v7_<tag>.{csv,txt}   (test sets)
  results_v7/full_result/aer_vs_v7_demo_mixed_ext.{csv,txt}

Final section: V4 vs V6 vs V7 comparison table across all test sets + demo_mixed.
"""
import sys, os
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader

sys.path.append(os.getcwd() + '/model')
sys.path.append(os.getcwd() + '/data_preparation')

from data_preparation import execution, helper

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("results_v7/full_result/model.pth")
STATS_PATH = Path("results_v7/full_result/standardization_stats.pth")
OUT_DIR    = Path("results_v7/full_result")
CLOSE_THRESH = 15.0   # % relative error

TEST_SETS = [
    ("test_50",  Path("datav1/test_50_final.csv")),
    ("test_100", Path("datav1/test_100_final.csv")),
    ("test_200", Path("datav1/test_200_final.csv")),
    ("test_300", Path("datav1/test_300_final.csv")),
]

DEMO_CIRCUIT_DIR = Path("datav1/demo_mixed")
SHOTS = 1024

# ── Load V7 model ──────────────────────────────────────────────────────────────
print("Loading V7 model + stats …")
import builder
from torchpack.utils.config import configs as tconfigs

tconfigs.load("model/parameter/default/config.yaml", recursive=True)
tconfigs.parameter_name = "default"
tconfigs.model.num_layers = 4
tconfigs.model.dropout    = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
model  = builder.make_model().to(device)
model.load_state_dict(torch.load(str(MODEL_PATH), weights_only=True))
model.eval()

saved = torch.load(str(STATS_PATH), weights_only=False)
print("  V7 model loaded (4 layers, dropout=0.1)\n")

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_nqubits(name):
    for p in name.split("_"):
        try: return int(p)
        except ValueError: pass
    return 0

def get_tier(nq):
    if nq <= 6:  return "LOW"
    if nq <= 14: return "MED"
    return "HIGH"

SEP = "-" * 74

def _predict(df_feat):
    """Extract features and predict. Returns y_pred_log array."""
    sample, time_taken = execution.Execution().generate_training_sample_execution_time(df_feat)
    refined, _  = helper.refine_training_data(sample, non_zero_indices=saved["non_zero_indices"])
    std_data, _ = helper.standardization_training_data(refined, stats=saved["stats"])
    for i in range(len(std_data)):
        std_data[i].y = float(np.log(float(time_taken[i]) + 1e-8))
        if std_data[i].global_features.dim() == 1:
            std_data[i].global_features = std_data[i].global_features.unsqueeze(0)
    loader = DataLoader(std_data, batch_size=64, shuffle=False)
    y_pred_log = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            y_pred_log.extend(np.atleast_1d(out.cpu().numpy()).tolist())
    return np.array(y_pred_log)

def _metrics(log_a, log_p):
    ss_res = np.sum((log_a - log_p) ** 2)
    ss_tot = np.sum((log_a - log_a.mean()) ** 2)
    r2   = 1 - ss_res / (ss_tot + 1e-12)
    mse  = np.mean((log_a - log_p) ** 2)
    nmse = mse / (np.var(log_a) + 1e-12)
    return r2, mse, nmse

def _tier_block_lines(sub_df, tier_key, tier_range):
    """Return list of text lines for one tier block."""
    lines = [
        f"  ──── {tier_key}  ({tier_range}) ────",
        f"  {'Circuit':<47} {'Actual':>9}  {'Predicted':>9}  ",
        SEP,
    ]
    for _, r in sub_df.iterrows():
        act_s  = f"{r['actual_ms']:>7.2f}ms" if pd.notna(r.get("actual_ms")) and r.get("actual_ms", 0) > 0 else "    OOM  "
        pred_s = f"{r['pred_ms']:>7.2f}ms"   if pd.notna(r.get("pred_ms"))   else "      —  "
        if pd.notna(r.get("err_pct")):
            mark = "✓" if r["within_15"] else "✗"
        else:
            mark = "N/A"
        note = f"  [{r['note']}]" if r.get("note") else ""
        lines.append(f"  {r['circuit']:<47} {act_s}  {pred_s}  {mark}{note}")
    lines.append(SEP)

    # Metrics for this tier
    valid = sub_df[sub_df["actual_ms"].notna() & (sub_df["actual_ms"] > 0) & sub_df["pred_ms"].notna()]
    nok   = int(sub_df["within_15"].sum())
    lines.append(f"  ─── Metrics — {tier_key}  ({tier_range})")
    if len(valid) >= 2:
        la, lp = np.log(valid["actual_ms"].values / 1000), np.log(valid["pred_ms"].values / 1000)
        r2, mse, nmse = _metrics(la, lp)
        mae  = np.mean(np.abs(valid["actual_ms"].values - valid["pred_ms"].values))
        max_ = np.max(np.abs(valid["actual_ms"].values - valid["pred_ms"].values))
        lines += [
            f"  R²  (log)      : {r2:.4f}",
            f"  MSE (log)      : {mse:.6f}",
            f"  NMSE (log)     : {nmse:.6f}",
            f"  Mean abs error : {mae:.3f} ms",
            f"  Max  abs error : {max_:.3f} ms",
        ]
    lines.append(f"  ✓ within 15%   : {nok}/{len(sub_df)} circuits")
    lines.append("")
    return lines

def _save_results(df, tag, header_label, tier_defs, overall_filter=None):
    """Save CSV and TXT for a test run. Returns tier-level metric dict."""
    csv_out = OUT_DIR / f"aer_vs_v7_{tag}.csv"
    df.to_csv(csv_out, index=False)

    txt_lines = [
        "",
        "=" * 75,
        f"  {header_label}",
        "=" * 75,
        "",
    ]
    tier_metrics_out = {}
    for tier_key, tier_range in tier_defs:
        sub = df[df["tier"] == tier_key]
        if len(sub) == 0:
            continue
        txt_lines += _tier_block_lines(sub, tier_key, tier_range)
        valid = sub[sub["actual_ms"].notna() & (sub["actual_ms"] > 0) & sub["pred_ms"].notna()]
        if len(valid) >= 2:
            la, lp = np.log(valid["actual_ms"].values / 1000), np.log(valid["pred_ms"].values / 1000)
            r2, mse, nmse = _metrics(la, lp)
            tier_metrics_out[tier_key] = {
                "r2": r2, "mse": mse, "nmse": nmse,
                "nok": int(sub["within_15"].sum()), "n": len(sub),
            }

    # Overall (optionally filtered to in-distribution only)
    df_ov = df if overall_filter is None else df[df["tier"].isin(overall_filter)]
    val_ov = df_ov[df_ov["actual_ms"].notna() & (df_ov["actual_ms"] > 0) & df_ov["pred_ms"].notna()]
    nok_ov = int(df_ov["within_15"].sum())

    txt_lines.append("=" * 75)
    txt_lines.append(f"  ─── Metrics — Overall")
    if len(val_ov) >= 2:
        la, lp = np.log(val_ov["actual_ms"].values / 1000), np.log(val_ov["pred_ms"].values / 1000)
        r2_ov, mse_ov, nmse_ov = _metrics(la, lp)
        mae_ov = np.mean(np.abs(val_ov["actual_ms"].values - val_ov["pred_ms"].values))
        max_ov = np.max(np.abs(val_ov["actual_ms"].values - val_ov["pred_ms"].values))
        txt_lines += [
            f"  R²  (log)      : {r2_ov:.4f}",
            f"  MSE (log)      : {mse_ov:.6f}",
            f"  NMSE (log)     : {nmse_ov:.6f}",
            f"  Mean abs error : {mae_ov:.3f} ms",
            f"  Max  abs error : {max_ov:.3f} ms",
        ]
        tier_metrics_out["overall"] = {
            "r2": r2_ov, "mse": mse_ov, "nmse": nmse_ov,
            "nok": nok_ov, "n": len(df_ov),
        }
    txt_lines.append(f"  ✓ within 15%   : {nok_ov}/{len(df_ov)} circuits")
    txt_lines.append("=" * 75)
    txt_lines.append("")

    txt_out = OUT_DIR / f"aer_vs_v7_{tag}.txt"
    txt_out.write_text("\n".join(txt_lines))
    print(f"  CSV → {csv_out}")
    print(f"  TXT → {txt_out}")
    return tier_metrics_out


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — External test sets (test_50 / test_100 / test_200 / test_300)
# ══════════════════════════════════════════════════════════════════════════════
all_summary = {}   # tag -> tier_metrics dict

for tag, csv_path in TEST_SETS:
    print(f"\n{'='*60}")
    print(f"  {tag.upper()}  ({csv_path})")
    print(f"{'='*60}")

    df_csv = pd.read_csv(csv_path)
    print(f"  {len(df_csv)} circuits loaded.", flush=True)

    df_feat = df_csv[["quantum_circuit", "time_taken"]].copy()
    df_feat["device"] = "aer"

    y_pred_log = _predict(df_feat)

    pred_ms   = np.exp(y_pred_log) * 1000
    actual_ms = df_csv["time_taken"].values[:len(pred_ms)] * 1000
    err_pct   = np.abs(pred_ms - actual_ms) / (actual_ms + 1e-10) * 100
    within    = err_pct <= CLOSE_THRESH

    names   = df_csv["circuit_name"].values[:len(pred_ms)]
    nqubits = [get_nqubits(n) for n in names]
    tiers   = [get_tier(q) for q in nqubits]

    rows = [{"circuit": names[i], "n_qubits": nqubits[i], "tier": tiers[i],
             "actual_ms": actual_ms[i], "pred_ms": pred_ms[i],
             "err_pct": err_pct[i], "within_15": bool(within[i]), "note": ""}
            for i in range(len(pred_ms))]
    df = pd.DataFrame(rows).sort_values(["tier", "n_qubits", "circuit"]).reset_index(drop=True)

    # Quick console summary
    for tk in ["LOW", "MED", "HIGH"]:
        sub = df[df["tier"] == tk]
        if len(sub) == 0: continue
        valid = sub[sub["actual_ms"].notna()]
        if len(valid) >= 2:
            la, lp = np.log(valid["actual_ms"].values/1000), np.log(valid["pred_ms"].values/1000)
            r2, _, _ = _metrics(la, lp)
            print(f"  {tk:6s}  R²={r2:.4f}  ✓={int(sub['within_15'].sum())}/{len(sub)}", flush=True)
    la_all = np.log(df["actual_ms"].values/1000); lp_all = np.log(df["pred_ms"].values/1000)
    r2_all, mse_all, _ = _metrics(la_all, lp_all)
    print(f"  Overall R²={r2_all:.4f}  MSE={mse_all:.4f}  ✓={int(df['within_15'].sum())}/{len(df)}", flush=True)

    tier_defs = [("LOW","2–6 qubits"), ("MED","7–14 qubits"), ("HIGH","15+ qubits")]
    label = f"{tag.upper()}  ({len(df)} circuits)  — Aer vs V7"
    all_summary[tag] = _save_results(df, tag, label, tier_defs)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Demo-mixed circuits (Aer + V7 prediction)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  DEMO_MIXED  — running AerSimulator + V7 prediction")
print(f"{'='*60}")

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
simulator_dm = AerSimulator()

CIRCUIT_LIST = [
    # LOW (2–6q)
    ("low_q02_realamp_reps3_s0",    2,  "LOW"),
    ("low_q03_su2_reps3_s1",        3,  "LOW"),
    ("low_q04_twolocal_reps4_s2",   4,  "LOW"),
    ("low_q04_realamp_reps5_s3",    4,  "LOW"),
    ("low_q05_su2_reps4_s4",        5,  "LOW"),
    ("low_q05_twolocal_reps5_s5",   5,  "LOW"),
    ("low_q06_realamp_reps4_s6",    6,  "LOW"),
    ("low_q06_su2_reps5_s7",        6,  "LOW"),
    ("low_q06_twolocal_reps6_s8",   6,  "LOW"),
    ("low_q06_realamp_reps8_s9",    6,  "LOW"),
    # MED (8–14q)
    ("med_q08_realamp_reps3_s10",   8,  "MED"),
    ("med_q09_su2_reps3_s11",       9,  "MED"),
    ("med_q10_twolocal_reps3_s12",  10, "MED"),
    ("med_q10_realamp_reps4_s13",   10, "MED"),
    ("med_q11_su2_reps4_s14",       11, "MED"),
    ("med_q12_twolocal_reps4_s15",  12, "MED"),
    ("med_q12_realamp_reps5_s16",   12, "MED"),
    ("med_q13_su2_reps4_s17",       13, "MED"),
    ("med_q14_twolocal_reps5_s18",  14, "MED"),
    ("med_q14_realamp_reps6_s19",   14, "MED"),
    # HIGH (15–30q)
    ("high_q15_realamp_reps3_s20",  15, "HIGH"),
    ("high_q16_su2_reps3_s21",      16, "HIGH"),
    ("high_q17_twolocal_reps3_s22", 17, "HIGH"),
    ("high_q18_realamp_reps4_s23",  18, "HIGH"),
    ("high_q19_su2_reps4_s24",      19, "HIGH"),
    ("high_q20_twolocal_reps4_s25", 20, "HIGH"),
    ("high_q21_realamp_reps3_s100", 21, "HIGH"),
    ("high_q21_realamp_reps3_s26",  21, "HIGH"),
    ("high_q22_su2_reps3_s101",     22, "HIGH"),
    ("high_q22_su2_reps3_s27",      22, "HIGH"),
    ("high_q23_twolocal_reps2_s102",23, "HIGH"),
    ("high_q24_realamp_reps3_s103", 24, "HIGH"),
    ("high_q24_twolocal_reps3_s28", 24, "HIGH"),
    ("high_q25_realamp_reps3_s29",  25, "HIGH"),
    ("high_q25_su2_reps3_s104",     25, "HIGH"),
    ("high_q26_realamp_reps2_s105", 26, "HIGH"),
    ("high_q27_su2_reps2_s106",     27, "HIGH"),
    ("high_q28_realamp_reps2_s107", 28, "HIGH"),
    ("high_q29_su2_reps2_s108",     29, "HIGH"),
    ("high_q30_realamp_reps1_s109", 30, "HIGH"),
    # EXTRAP (31–32q)
    ("high_q31_realamp_reps1_s200", 31, "EXTRAP"),
    ("high_q32_su2_reps1_s201",     32, "EXTRAP"),
]

aer_rows    = []
all_results = []

for name, nq, tier in CIRCUIT_LIST:
    qasm_path = DEMO_CIRCUIT_DIR / f"{name}.qasm"
    if not qasm_path.exists():
        print(f"  MISSING: {name}", flush=True)
        all_results.append({"name": name, "nq": nq, "tier": tier,
                             "actual_ms": None, "qasm": None, "note": "missing"})
        continue

    qasm_str = qasm_path.read_text()
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    qc_run = qc.copy()
    if len(qc_run.clbits) == 0:
        qc_run.measure_all()

    times = []
    note  = ""
    for _ in range(3):
        try:
            result = simulator_dm.run(qc_run, shots=SHOTS).result()
            t = result.results[0].time_taken
            if t > 0:
                times.append(t)
            else:
                note = "OOM"; break
        except Exception as e:
            note = "Aer error"; break

    if len(times) == 3:
        actual_ms_val = float(np.mean(times)) * 1000
        aer_rows.append({"circuit_name": name, "quantum_circuit": qasm_str,
                         "time_taken": float(np.mean(times)), "tier": tier, "n_qubits": nq})
        print(f"  {name:<47}  {actual_ms_val:>10.2f} ms", flush=True)
    elif len(times) > 0:
        actual_ms_val = float(np.mean(times)) * 1000
        note = f"partial {len(times)}/3"
        aer_rows.append({"circuit_name": name, "quantum_circuit": qasm_str,
                         "time_taken": float(np.mean(times)), "tier": tier, "n_qubits": nq})
        print(f"  {name:<47}  {actual_ms_val:>10.2f} ms  [{note}]", flush=True)
    else:
        actual_ms_val = None
        print(f"  {name:<47}  {'OOM/failed':>10}       [{note}]", flush=True)

    all_results.append({"name": name, "nq": nq, "tier": tier,
                        "actual_ms": actual_ms_val, "qasm": qasm_str, "note": note})

print(f"\n  {len(aer_rows)}/{len(CIRCUIT_LIST)} circuits ran on Aer\n", flush=True)

# Predict with V7
df_aer = pd.DataFrame(aer_rows)[["quantum_circuit", "time_taken"]].copy()
df_aer["device"] = "aer"
y_pred_log_dm = _predict(df_aer)
pred_ms_list  = [float(np.exp(lp) * 1000) for lp in y_pred_log_dm]
name_to_pred  = {row["circuit_name"]: pred_ms_list[i] for i, row in enumerate(aer_rows)}

# Build final dataframe
dm_rows = []
for r in all_results:
    pred_ms_val   = name_to_pred.get(r["name"])
    actual_ms_val = r["actual_ms"]
    if actual_ms_val and actual_ms_val > 0 and pred_ms_val:
        err_pct = abs(pred_ms_val - actual_ms_val) / actual_ms_val * 100
        within  = err_pct <= CLOSE_THRESH
    else:
        err_pct = None
        within  = False
    dm_rows.append({
        "circuit":   r["name"],  "n_qubits": r["nq"],  "tier": r["tier"],
        "actual_ms": actual_ms_val, "pred_ms": pred_ms_val,
        "err_pct":   err_pct,    "within_15": within,  "note": r["note"],
    })
df_dm = pd.DataFrame(dm_rows)

# Console summary
for tk in ["LOW", "MED", "HIGH", "EXTRAP"]:
    sub = df_dm[df_dm["tier"] == tk]
    if len(sub) == 0: continue
    valid = sub[sub["actual_ms"].notna() & (sub["actual_ms"] > 0) & sub["pred_ms"].notna()]
    nok   = int(sub["within_15"].sum())
    if len(valid) >= 2:
        la, lp = np.log(valid["actual_ms"].values/1000), np.log(valid["pred_ms"].values/1000)
        r2, _, _ = _metrics(la, lp)
        print(f"  {tk:8s}  R²={r2:.4f}  ✓={nok}/{len(sub)}", flush=True)
    else:
        print(f"  {tk:8s}  ✓={nok}/{len(sub)}", flush=True)

tier_defs_dm = [
    ("LOW",    "2–6 qubits"),
    ("MED",    "7–14 qubits"),
    ("HIGH",   "15–30 qubits  ← training range"),
    ("EXTRAP", "31–32 qubits  ← out of distribution"),
]
label_dm = "DEMO_MIXED + EXTRAPOLATION  (42 circuits, 2–32q)  — Aer vs V7"
all_summary["demo_mixed"] = _save_results(
    df_dm, "demo_mixed_ext", label_dm, tier_defs_dm,
    overall_filter=["LOW", "MED", "HIGH"]   # in-distribution only for overall
)


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Comparison: V4 vs V6 vs V7
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*80)
print("  COMPARISON — V4  vs  V6  vs  V7")
print("="*80)

def load_csv_stats(csv_path, pred_col="pred_ms", tier_col="tier"):
    """Load any version's CSV and return tier-level metric dict."""
    p = Path(csv_path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # Normalise column name differences between versions
    if "predicted_ms" in df.columns and pred_col not in df.columns:
        df = df.rename(columns={"predicted_ms": pred_col})
    if pred_col not in df.columns:
        return None
    # Build tier if missing
    if tier_col not in df.columns:
        df[tier_col] = df["circuit"].apply(lambda n: get_tier(get_nqubits(n)))
    # Filter extrap
    if "EXTRAP" in df[tier_col].values:
        df = df[df[tier_col] != "EXTRAP"].copy()
    # Build within_15 if missing
    if "within_15" not in df.columns:
        err = np.abs(df[pred_col].values - df["actual_ms"].values) / (df["actual_ms"].values + 1e-10) * 100
        df["within_15"] = err <= CLOSE_THRESH
    df = df[df["actual_ms"].notna() & (df["actual_ms"] > 0) & df[pred_col].notna()]
    if len(df) == 0:
        return None
    out = {}
    la_all = np.log(df["actual_ms"].values / 1000)
    lp_all = np.log(df[pred_col].values    / 1000)
    r2_all, mse_all, nmse_all = _metrics(la_all, lp_all)
    out["overall"] = {"r2": r2_all, "mse": mse_all, "nmse": nmse_all,
                      "nok": int(df["within_15"].sum()), "n": len(df)}
    for tk in ["LOW", "MED", "HIGH"]:
        sub = df[df[tier_col] == tk]
        if len(sub) == 0:
            out[tk] = None; continue
        la2 = np.log(sub["actual_ms"].values / 1000)
        lp2 = np.log(sub[pred_col].values    / 1000)
        r2t, _, _ = _metrics(la2, lp2)
        out[tk] = {"r2": r2t, "nok": int(sub["within_15"].sum()), "n": len(sub)}
    return out

def _r(d, k):
    if d is None or d.get(k) is None: return "     N/A"
    return f"{d[k]['r2']:>8.4f}"

def _ok(d, k):
    if d is None or d.get(k) is None: return "     N/A"
    return f"{d[k]['nok']:>3}/{d[k]['n']:>3} ✓"

for tag, _ in TEST_SETS:
    v4 = load_csv_stats(f"results_v4/full_result/aer_vs_v4_{tag}.csv")
    v6 = load_csv_stats(f"results_v6/full_result/aer_vs_v6_{tag}.csv")
    v7 = all_summary.get(tag)

    print(f"\n  ── {tag.upper()} ──")
    print(f"  {'':12}  {'V4':>14}  {'V6':>14}  {'V7':>14}")
    print("  " + "-"*60)
    for tk in ["LOW", "MED", "HIGH", "overall"]:
        print(f"  R² {tk:<9}  {_r(v4,tk):>14}  {_r(v6,tk):>14}  {_r(v7,tk):>14}")
    print(f"  {'✓ 15%':<12}  {_ok(v4,'overall'):>14}  {_ok(v6,'overall'):>14}  {_ok(v7,'overall'):>14}")

# Demo-mixed
v4_dm = None
v6_dm = load_csv_stats("results_v6/full_result/aer_vs_v6_demo_mixed.csv")
v7_dm = all_summary.get("demo_mixed")

print(f"\n  ── DEMO_MIXED (2–30q in-distribution) ──")
print(f"  {'':12}  {'V4':>14}  {'V6':>14}  {'V7':>14}")
print("  " + "-"*60)
for tk in ["LOW", "MED", "HIGH", "overall"]:
    print(f"  R² {tk:<9}  {_r(v4_dm,tk):>14}  {_r(v6_dm,tk):>14}  {_r(v7_dm,tk):>14}")
print(f"  {'✓ 15%':<12}  {'N/A':>14}  {_ok(v6_dm,'overall'):>14}  {_ok(v7_dm,'overall'):>14}")

print("\n" + "="*80)
print("  V4  = 3 layers, dropout=0.0  (smaller dataset, max 20q)")
print("  V6  = 3 layers, dropout=0.1  (combined dataset, max 30q)")
print("  V7  = 4 layers, dropout=0.1  (combined dataset, max 32q)  ← current")
print("="*80)
print("\nAll done.")
