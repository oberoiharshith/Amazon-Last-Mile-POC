"""
CLI entrypoint to run baseline vs ML-aware dispatch experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.simulation_engine import SimConfig
from src.dispatch_optimizer import baseline_fifo, ml_aware_dispatch
from src.experiment_framework import run_experiment


def _sanitize_for_json(obj):
    """
    Make output JSON strictly valid:
    - convert NaN/Inf to None (JSON null)
    - recursively sanitize dicts/lists
    """
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _save_plots(runs: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    for metric, ylabel in [
        ("on_time_rate", "On-time rate"),
        ("avg_lead_time_min", "Avg lead time (min)"),
        ("cost_per_pkg", "Cost per package"),
    ]:
        plt.figure()
        for variant in ["baseline", "treatment"]:
            sub = runs[runs["variant"] == variant].sort_values("day")
            plt.plot(sub["day"], sub[metric], label=variant)
        plt.xlabel("Day")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}_by_day.png"))
        plt.close()

    plt.figure()
    agg = runs.groupby("variant")[["on_time_rate", "avg_lead_time_min", "cost_per_pkg"]].mean()
    agg["on_time_rate"].plot(kind="bar")
    plt.ylabel("On-time rate (avg)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "on_time_rate_avg.png"))
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--drivers", type=int, default=60)
    ap.add_argument("--stations", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)

    # Optional knobs to make the system “harder” (more delays) so on_time_rate moves
    ap.add_argument("--sla_minutes", type=int, default=None, help="Override SLA minutes (e.g., 180)")
    ap.add_argument("--traffic_sigma", type=float, default=None, help="Override traffic noise (e.g., 0.45)")

    args = ap.parse_args()

    cfg = SimConfig(
        n_drivers=args.drivers,
        n_stations=args.stations,
        seed=args.seed,
    )

    # Apply optional overrides
    if args.sla_minutes is not None:
        cfg.sla_minutes = int(args.sla_minutes)
    if args.traffic_sigma is not None:
        cfg.traffic_sigma = float(args.traffic_sigma)

    result = run_experiment(
        days=args.days,
        cfg=cfg,
        baseline_policy=baseline_fifo,
        treatment_policy=ml_aware_dispatch,
    )

    reports_dir = Path("reports")
    plots_dir = reports_dir / "plots"
    reports_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    result.runs.to_csv(reports_dir / "results_runs.csv", index=False)

    # Strict JSON (no NaN/Inf)
    safe_summary = _sanitize_for_json(result.summary)
    with open(reports_dir / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(safe_summary, f, indent=2)

    _save_plots(result.runs, str(plots_dir))

    print("Wrote:")
    print(f"- {reports_dir / 'results_runs.csv'}")
    print(f"- {reports_dir / 'results_summary.json'}")
    print(f"- {plots_dir}/*.png")


if __name__ == "__main__":
    main()