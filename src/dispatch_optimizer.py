"""
Dispatch policies:
- baseline: FIFO within station
- ml_aware: prioritize low-risk assignments for tight SLA and push high-risk earlier
  (simple heuristic: sort by predicted risk descending so risky orders get earlier capacity)
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def baseline_fifo(pending_df: pd.DataFrame, state: Dict) -> Dict[int, List[int]]:
    """
    FIFO: assign earliest-arrived orders first, per station, spreading across drivers.
    """
    cfg = state["cfg"]
    t = state["t"]
    driver_station = state["driver_station"]
    driver_load = state["driver_load"]

    # group by station
    assignments: Dict[int, List[int]] = {d: [] for d in driver_station.keys()}

    # order by arrival time
    pending_df = pending_df.sort_values(["order_time_minute"])

    # list drivers per station with remaining capacity
    drivers_by_station: Dict[int, List[int]] = {s: [] for s in range(cfg.n_stations)}
    for d, s in driver_station.items():
        if driver_load[d] < cfg.driver_capacity:
            drivers_by_station[s].append(d)

    for s in range(cfg.n_stations):
        drivers_by_station[s].sort()

    # round-robin fill
    rr_pos = {s: 0 for s in range(cfg.n_stations)}
    for ix, row in pending_df.iterrows():
        s = int(row["station_id"])
        if not drivers_by_station[s]:
            continue
        d = drivers_by_station[s][rr_pos[s] % len(drivers_by_station[s])]
        assignments[d].append(int(ix))
        rr_pos[s] += 1

    # prune empty
    return {d: v for d, v in assignments.items() if v}


def ml_aware_dispatch(pending_df: pd.DataFrame, state: Dict) -> Dict[int, List[int]]:
    """
    ML-aware dispatch using a blended priority:
      priority = 0.7 * delay_risk + 0.3 * normalized_wait_time

    This avoids the "all risky first" behavior that can worsen average flow time
    while still pulling forward orders likely to miss SLA.
    """
    cfg = state["cfg"]
    t = state["t"]
    driver_station = state["driver_station"]
    driver_load = state["driver_load"]
    delay_proba: Dict[int, float] = state.get("delay_proba", {})

    tmp = pending_df.copy()

    # predicted delay risk for each order index
    tmp["delay_proba"] = [float(delay_proba.get(int(ix), 0.5)) for ix in tmp.index]

    # how long the order has been waiting so far
    tmp["waiting_time"] = (t - tmp["order_time_minute"]).clip(lower=0)

    # normalize waiting_time to 0..1 to keep scale stable
    wt_max = float(tmp["waiting_time"].max()) if len(tmp) else 0.0
    if wt_max <= 0.0:
        tmp["waiting_norm"] = 0.0
    else:
        tmp["waiting_norm"] = tmp["waiting_time"] / wt_max

    # blended priority score
    tmp["priority_score"] = 0.7 * tmp["delay_proba"] + 0.3 * tmp["waiting_norm"]

    # highest priority first; tie-break by earlier arrival
    tmp = tmp.sort_values(["priority_score", "order_time_minute"], ascending=[False, True])

    assignments: Dict[int, List[int]] = {d: [] for d in driver_station.keys()}

    # drivers available per station
    drivers_by_station: Dict[int, List[int]] = {s: [] for s in range(cfg.n_stations)}
    for d, s in driver_station.items():
        if driver_load[d] < cfg.driver_capacity:
            drivers_by_station[s].append(d)
    for s in range(cfg.n_stations):
        drivers_by_station[s].sort()

    # round-robin fill within station
    rr_pos = {s: 0 for s in range(cfg.n_stations)}
    for ix, row in tmp.iterrows():
        s = int(row["station_id"])
        if not drivers_by_station[s]:
            continue
        d = drivers_by_station[s][rr_pos[s] % len(drivers_by_station[s])]
        assignments[d].append(int(ix))
        rr_pos[s] += 1

    return {d: v for d, v in assignments.items() if v}
