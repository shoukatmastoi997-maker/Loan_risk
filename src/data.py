from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_openml

from .config import RAW_DIR, TARGET_COL
from .utils import log

RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_data(refresh: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Returns X, y, full dataframe (including target).
    Downloads data from OpenML if not cached.
    """
    cache_path = RAW_DIR / "credit_g.csv"

    if cache_path.exists() and not refresh:
        log(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path)
    else:
        log("Downloading credit-g dataset from OpenML...")
        data = fetch_openml("credit-g", version=1, as_frame=True)
        df = data.frame
        log(f"Saving cache to {cache_path}")
        df.to_csv(cache_path, index=False)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y, df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET_COL]
