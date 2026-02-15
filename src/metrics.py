"""
KPI calculations for last-mile simulation outputs.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd


def compute_kpis(orders: pd.DataFrame, day_summary: Dict, sla_minutes: int) -> Dict[str, float]:
    lead_time = orders["delivered_time_minute"] - orders["order_time_minute"]
    on_time = (lead_time <= sla_minutes).astype(int)

    assigned_rate = float((orders["assigned_driver"] >= 0).mean())
    on_time_rate = float(on_time.mean())
    avg_lead_time = float(lead_time.mean())
    p90_lead_time = float(np.percentile(lead_time, 90))
    cost_orders = float(orders["cost"].sum())
    overtime_cost = float(day_summary.get("overtime_cost_total", 0.0))
    total_cost = cost_orders + overtime_cost
    cost_per_pkg = float(total_cost / max(1, len(orders)))

    # utilization proxy: avg active minutes / shift minutes
    active = day_summary.get("driver_active_minutes", {})
    if active:
        util = float(np.mean([v for v in active.values()]) / max(1.0, float(sla_minutes)))
    else:
        util = float("nan")

    return {
        "n_orders": int(len(orders)),
        "assigned_rate": assigned_rate,
        "on_time_rate": on_time_rate,
        "avg_lead_time_min": avg_lead_time,
        "p90_lead_time_min": p90_lead_time,
        "total_cost": total_cost,
        "cost_per_pkg": cost_per_pkg,
        "overtime_cost_total": overtime_cost,
        "utilization_proxy": util,
    }
