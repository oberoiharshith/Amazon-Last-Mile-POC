"""
Experiment runner: compares baseline vs ML-aware dispatch across multiple days and seeds.
Outputs summary + per-run metrics and simple significance tests.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Callable, Optional, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from .simulation_engine import SimConfig, generate_synthetic_orders, simulate_day
from .feature_engineering import add_driver_load_feature, build_model_matrix
from .delay_model import train_delay_model, predict_delay_proba, ModelArtifacts
from .metrics import compute_kpis


@dataclass
class ExperimentResult:
    runs: pd.DataFrame
    summary: Dict


def _sig_test(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    # paired t-test where possible
    if len(a) != len(b) or len(a) < 2:
        return {"p_value": float("nan"), "t_stat": float("nan")}
    t_stat, p = stats.ttest_rel(b, a, nan_policy="omit")
    return {"p_value": float(p), "t_stat": float(t_stat)}


def run_experiment(
    days: int,
    cfg: SimConfig,
    baseline_policy: Callable,
    treatment_policy: Callable,
) -> ExperimentResult:
    # 1) generate orders
    orders = generate_synthetic_orders(days, cfg)

    # 2) baseline sim to produce training labels with realistic assignment times
    baseline_outputs = []
    baseline_day_summaries = []
    for day in range(days):
        od = orders[orders["day"] == day].copy()
        out, summ = simulate_day(od, cfg, baseline_policy)
        baseline_outputs.append(out)
        baseline_day_summaries.append(summ)
    baseline_all = pd.concat(baseline_outputs, axis=0)

    # 3) train delay model on baseline outcomes
    baseline_all = add_driver_load_feature(baseline_all)
    X, y = build_model_matrix(baseline_all)
    artifacts = train_delay_model(baseline_all, X, y, seed=cfg.seed)

    # 4) run baseline + treatment on the same underlying orders, per day
    runs = []
    for day in range(days):
        od = orders[orders["day"] == day].copy()

        # baseline
        out_b, summ_b = simulate_day(od, cfg, baseline_policy)
        kpi_b = compute_kpis(out_b, summ_b, cfg.sla_minutes)

        # treatment: we need per-order delay proba before dispatch.
        # We approximate this by using the order-level features (and a placeholder driver_load).
        # In a real system you'd update features online; here we keep it simple but consistent.
        tmp = od.copy()
        tmp["assigned_driver"] = -1
        tmp["assigned_time_minute"] = np.nan
        tmp["delivered_time_minute"] = np.nan
        tmp["delay_flag"] = 0
        tmp["driver_load_at_assign"] = 0  # unknown pre-assign; proxy
        X_tmp = tmp[artifacts.feature_names].copy().fillna(0.0)
        proba = predict_delay_proba(artifacts.model, X_tmp)
        delay_map = {int(ix): float(p) for ix, p in zip(tmp.index, proba)}

        policy_state = {"delay_proba": delay_map}
        out_t, summ_t = simulate_day(od, cfg, treatment_policy, policy_state=policy_state)
        out_t = add_driver_load_feature(out_t)
        kpi_t = compute_kpis(out_t, summ_t, cfg.sla_minutes)

        runs.append({
            "day": day,
            "variant": "baseline",
            **kpi_b,
        })
        runs.append({
            "day": day,
            "variant": "treatment",
            **kpi_t,
        })

    runs_df = pd.DataFrame(runs)

    # 5) summary + stats
    pivot = runs_df.pivot_table(index="day", columns="variant", values=["on_time_rate", "avg_lead_time_min", "cost_per_pkg"])
    # flatten for easier use
    on_a = pivot["on_time_rate"]["baseline"].values
    on_b = pivot["on_time_rate"]["treatment"].values
    lt_a = pivot["avg_lead_time_min"]["baseline"].values
    lt_b = pivot["avg_lead_time_min"]["treatment"].values
    cp_a = pivot["cost_per_pkg"]["baseline"].values
    cp_b = pivot["cost_per_pkg"]["treatment"].values

    summary = {
        "model_metrics": artifacts.metrics,
        "impact": {
            "on_time_rate_delta": float(np.nanmean(on_b - on_a)),
            "avg_lead_time_delta_min": float(np.nanmean(lt_b - lt_a)),
            "cost_per_pkg_delta": float(np.nanmean(cp_b - cp_a)),
        },
        "significance": {
            "on_time_rate": _sig_test(on_a, on_b),
            "avg_lead_time_min": _sig_test(lt_a, lt_b),
            "cost_per_pkg": _sig_test(cp_a, cp_b),
        },
    }

    return ExperimentResult(runs=runs_df, summary=summary)
