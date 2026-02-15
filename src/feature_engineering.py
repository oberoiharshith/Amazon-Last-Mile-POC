"""
Feature engineering for delay prediction.
Keeps it simple and explicit to make the pipeline easy to review.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


FEATURE_COLS = [
    "distance_km",
    "traffic_index",
    "area_density",
    "weather_proxy",
    "time_bucket",
    "station_id",
    "driver_load_at_assign",
]


def add_driver_load_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximates driver load at assignment time. In this POC,
    we can compute it from (day, driver, assigned_time) ordering.
    """
    out = df.copy()
    out["driver_load_at_assign"] = 0

    assigned = out[out["assigned_driver"] >= 0].copy()
    assigned.sort_values(["day", "assigned_driver", "assigned_time_minute"], inplace=True)

    # cumulative count per driver per day before current assignment
    assigned["driver_load_at_assign"] = (
        assigned.groupby(["day", "assigned_driver"]).cumcount()
    )

    out.loc[assigned.index, "driver_load_at_assign"] = assigned["driver_load_at_assign"].values
    return out


def build_model_matrix(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    # basic cleanup
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["delay_flag"].astype(int).values
    return X, y
