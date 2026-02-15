"""
Discrete Event Simulation for a simplified last-mile system.

Core objects:
- Orders arrive over time at stations
- Drivers have capacity and shifts
- Dispatch assigns orders to drivers
- Travel time depends on distance + stochastic traffic
- Outcomes: delivery time, delay flag, cost, utilization

This is intentionally simplified but structured like a real simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import pandas as pd


@dataclass
class SimConfig:
    n_stations: int = 6
    n_drivers: int = 60
    driver_capacity: int = 30  # max packages per driver per day
    shift_minutes: int = 10 * 60  # 10 hours
    sla_minutes: int = 180 # 3 hours target from order_time to delivered
    base_speed_kmph: float = 25.0
    traffic_sigma: float = 0.45  # multiplicative noise
    service_time_mean: float = 4.0  # minutes per stop
    service_time_sigma: float = 1.5
    dispatch_interval_minutes: int = 20
    seed: int = 7

    # cost model
    cost_per_km: float = 0.55
    overtime_cost_per_min: float = 0.35
    delay_penalty: float = 2.5  # per delayed package


def generate_synthetic_orders(days: int, cfg: SimConfig) -> pd.DataFrame:
    """
    Creates synthetic orders with:
    - station_id
    - order_time_minute (0..shift)
    - distance_km
    - time_window (minutes, optional)
    - covariates: traffic_index, area_density, weather_proxy, time_of_day_bucket
    """
    rng = np.random.default_rng(cfg.seed)

    rows = []
    for day in range(days):
        # daily volume: lognormal-ish
        daily_orders = int(rng.lognormal(mean=np.log(900), sigma=0.25))
        daily_orders = max(300, min(1800, daily_orders))

        station_id = rng.integers(0, cfg.n_stations, size=daily_orders)
        # order time: mixture (morning + afternoon)
        mix = rng.random(daily_orders)
        t = np.where(
            mix < 0.55,
            rng.normal(loc=180, scale=70, size=daily_orders),
            rng.normal(loc=420, scale=90, size=daily_orders),
        )
        order_time = np.clip(t, 0, cfg.shift_minutes - 1).astype(int)

        # distance: short-biased
        distance_km = np.clip(rng.gamma(shape=2.2, scale=2.0, size=daily_orders), 0.5, 25.0)

        # operational covariates
        traffic_index = np.clip(rng.normal(1.0, 0.15, size=daily_orders), 0.7, 1.6)
        # density: station-specific + noise
        station_density = rng.uniform(0.7, 1.4, size=cfg.n_stations)
        area_density = np.clip(station_density[station_id] + rng.normal(0, 0.08, size=daily_orders), 0.6, 1.6)
        weather_proxy = np.clip(rng.normal(0.0, 1.0, size=daily_orders), -2.5, 2.5)

        time_bucket = (order_time // 60).astype(int)  # hour bucket

        for i in range(daily_orders):
            rows.append({
                "day": day,
                "station_id": int(station_id[i]),
                "order_time_minute": int(order_time[i]),
                "distance_km": float(distance_km[i]),
                "traffic_index": float(traffic_index[i]),
                "area_density": float(area_density[i]),
                "weather_proxy": float(weather_proxy[i]),
                "time_bucket": int(time_bucket[i]),
            })

    return pd.DataFrame(rows)


def _travel_minutes(distance_km: float, traffic_multiplier: float, cfg: SimConfig, rng: np.random.Generator) -> float:
    # base travel time = distance / speed
    speed = cfg.base_speed_kmph / max(0.5, traffic_multiplier)
    minutes = (distance_km / speed) * 60.0
    # multiplicative traffic noise
    noise = rng.lognormal(mean=0.0, sigma=cfg.traffic_sigma)
    return minutes * noise


def simulate_day(
    orders_day: pd.DataFrame,
    cfg: SimConfig,
    dispatch_policy: Callable[[pd.DataFrame, Dict], Dict[int, List[int]]],
    policy_state: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Simulate a single day.

    dispatch_policy takes:
      - pending_orders dataframe (subset of orders_day not yet assigned)
      - current_state dict
    and returns a mapping: driver_id -> list of order indices (row indices from orders_day)
    """
    if policy_state is None:
        policy_state = {}

    rng = np.random.default_rng(cfg.seed + int(orders_day["day"].iloc[0]) * 1000 + 13)

    # drivers per station: roughly balanced
    drivers_per_station = cfg.n_drivers // cfg.n_stations
    driver_station = {d: d // drivers_per_station for d in range(cfg.n_drivers)}
    for d in range(cfg.n_drivers):
        driver_station[d] = min(driver_station[d], cfg.n_stations - 1)

    # driver availability
    driver_time = {d: 0.0 for d in range(cfg.n_drivers)}
    driver_load = {d: 0 for d in range(cfg.n_drivers)}
    driver_active_minutes = {d: 0.0 for d in range(cfg.n_drivers)}
    driver_km = {d: 0.0 for d in range(cfg.n_drivers)}

    orders_day = orders_day.copy()
    orders_day["assigned_driver"] = -1
    orders_day["assigned_time_minute"] = np.nan
    orders_day["delivered_time_minute"] = np.nan
    orders_day["delay_flag"] = 0
    orders_day["cost"] = 0.0

    # pending orders tracked by original index
    pending = orders_day.index.tolist()

    t = 0
    while t < cfg.shift_minutes:
        # new orders already included; dispatch on interval
        # pending orders that have arrived
        arrived_idx = [ix for ix in pending if orders_day.loc[ix, "order_time_minute"] <= t]

        if arrived_idx:
            pending_df = orders_day.loc[arrived_idx].copy()
            # state for policy
            state = {
                "t": t,
                "cfg": cfg,
                "driver_time": driver_time,
                "driver_load": driver_load,
                "driver_station": driver_station,
            }
            state.update(policy_state)

            assignments = dispatch_policy(pending_df, state)

            # apply assignments
            for driver_id, order_indices in assignments.items():
                if driver_id not in driver_time:
                    continue
                if driver_load[driver_id] >= cfg.driver_capacity:
                    continue
                # only allow matching station to keep it realistic
                ds = driver_station[driver_id]
                for ix in order_indices:
                    if ix not in pending:
                        continue
                    if orders_day.loc[ix, "station_id"] != ds:
                        continue
                    if orders_day.loc[ix, "order_time_minute"] > t:
                        continue
                    if driver_load[driver_id] >= cfg.driver_capacity:
                        break

                    # start service at max(current time, driver available time)
                    start = max(float(t), float(driver_time[driver_id]))
                    travel = _travel_minutes(
                        float(orders_day.loc[ix, "distance_km"]),
                        float(orders_day.loc[ix, "traffic_index"]),
                        cfg,
                        rng,
                    )
                    service = max(0.5, rng.normal(cfg.service_time_mean, cfg.service_time_sigma))
                    delivered = start + travel + service

                    orders_day.loc[ix, "assigned_driver"] = int(driver_id)
                    orders_day.loc[ix, "assigned_time_minute"] = float(start)
                    orders_day.loc[ix, "delivered_time_minute"] = float(delivered)

                    # update driver stats
                    driver_load[driver_id] += 1
                    driver_active_minutes[driver_id] += (travel + service)
                    driver_km[driver_id] += float(orders_day.loc[ix, "distance_km"])
                    driver_time[driver_id] = float(delivered)

                    pending.remove(ix)

        t += cfg.dispatch_interval_minutes

    # mark undelivered as delayed + expensive
    # (in real ops they'd roll, here we penalize)
    for ix in pending:
        orders_day.loc[ix, "assigned_driver"] = -1
        orders_day.loc[ix, "assigned_time_minute"] = np.nan
        orders_day.loc[ix, "delivered_time_minute"] = float(cfg.shift_minutes + cfg.sla_minutes)
        orders_day.loc[ix, "delay_flag"] = 1

    # compute delay flags + costs
    lead_time = orders_day["delivered_time_minute"] - orders_day["order_time_minute"]
    orders_day["delay_flag"] = (lead_time > cfg.sla_minutes).astype(int)

    # cost per order: distance cost + delay penalty; overtime cost applied per driver later
    orders_day["cost"] = orders_day["distance_km"] * cfg.cost_per_km + orders_day["delay_flag"] * cfg.delay_penalty

    # overtime cost
    overtime_total = 0.0
    for d in range(cfg.n_drivers):
        overtime = max(0.0, driver_time[d] - cfg.shift_minutes)
        overtime_total += overtime * cfg.overtime_cost_per_min

    day_summary = {
        "overtime_cost_total": float(overtime_total),
        "driver_active_minutes": driver_active_minutes,
        "driver_km": driver_km,
        "driver_load": driver_load,
    }

    return orders_day, day_summary
