"""
Delay risk model: train + evaluate.

We use GradientBoostingClassifier for portability (no extra deps).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass
class ModelArtifacts:
    model: object
    metrics: Dict[str, float]
    feature_names: list


def train_delay_model(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, seed: int = 7) -> ModelArtifacts:
    # time-based split by day to avoid leakage
    # hold out last 20% days
    days = sorted(df["day"].unique().tolist())
    cut = max(1, int(len(days) * 0.8))
    train_days = set(days[:cut])
    test_days = set(days[cut:])

    train_mask = df["day"].isin(train_days).values
    test_mask = df["day"].isin(test_days).values

    X_train, y_train = X.loc[train_mask], y[train_mask]
    X_test, y_test = X.loc[test_mask], y[test_mask]

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=220,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan"),
        "avg_precision": float(average_precision_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y_test, proba)),
        "test_delay_rate": float(y_test.mean()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    return ModelArtifacts(model=model, metrics=metrics, feature_names=list(X.columns))


def predict_delay_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
