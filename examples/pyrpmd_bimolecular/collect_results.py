"""汇总多个 PyRPMD rate 运行目录的 results/summary.json，输出 Markdown 表和 CSV。

用法: python collect_results.py runs/cmp/P1_200fs runs/cmp/P16_200fs ... [--csv out.csv]
只读取已保存的结果，不调用势能面。
"""
import argparse
import csv
import json
import math
from pathlib import Path

KB_HARTREE_PER_K = 3.166811563e-6
PAPER = {"qtst": 2.51e-16, "kappa": 0.776, "rate": 1.95e-16}  # CPC 184, 833 (2013), 300 K, P=128


def row(run: Path):
    d = json.loads((run / "results" / "summary.json").read_text())
    resolved = json.loads((run / "input" / "resolved.json").read_text())
    p = d["pmf"]
    sp = resolved.get("rate", {}).get("pmf", {}).get("sampling", {})
    ess = p.get("effective_samples_per_stream") or []
    ui = p.get("ui_comparison") or {}
    rate = d.get("rate_estimate_cm3_s")
    return {
        "run": run.name,
        "beads": resolved["beads"],
        "seed": resolved["seed"],
        "eq_prod_fs": f'{sp.get("equilibration_steps", 0) * 0.1:g}+{sp.get("production_steps", 0) * 0.1:g}',
        "xi_star": p.get("xi_star"),
        "barrier_mbar_hartree": p.get("free_energy_barrier_hartree"),
        "barrier_ui_hartree": ui.get("free_energy_barrier_hartree"),
        "barrier_ci95": p.get("barrier_percentile_95_ci_hartree"),
        "k_qtst_cm3_s": p.get("k_qtst_cm3_s"),
        "kappa": d.get("kappa"),
        "kappa_se": d.get("kappa_standard_error"),
        "plateau": d.get("plateau_detected"),
        "rate_cm3_s": rate,
        "rate_ci95": d.get("rate_percentile_95_ci_cm3_s"),
        "rate_over_paper": rate / PAPER["rate"] if rate else None,
        "median_ess_per_stream": sorted(ess)[len(ess) // 2] if ess else None,
        "max_endpoint_error_kbt": d.get("maximum_endpoint_energy_error_kbt"),
        "status": d.get("status"),
    }


def fmt(x, spec=".3g"):
    if x is None:
        return "—"
    if isinstance(x, list):
        return "[" + ", ".join(fmt(v, spec) for v in x) + "]"
    if isinstance(x, float):
        return format(x, spec)
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    rows = [row(r) for r in a.runs if (r / "results" / "summary.json").is_file()]
    cols = list(rows[0])
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    if a.csv:
        with a.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
