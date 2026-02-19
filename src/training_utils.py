"""Shared helpers for training/evaluation frame construction."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.preprocessing import CATEGORICAL_COLS, DATETIME_COLS, NUMERIC_COLS


def build_raw_from_ids(
    df_year_t: pd.DataFrame,
    ids: pd.Series,
    expected_raw_cols: Sequence[str],
    *,
    id_column: str = "RA",
) -> pd.DataFrame:
    """Build raw frame aligned to cohort ids and expected raw schema.

    - Preserves row order from `ids` via left merge.
    - Adds missing expected columns with dtype-safe nulls.
    - Drops extras and returns columns in the exact expected order.
    """
    if not isinstance(df_year_t, pd.DataFrame):
        raise TypeError(f"df_year_t must be pandas.DataFrame, got {type(df_year_t)}")
    if id_column not in df_year_t.columns:
        raise ValueError(f"ID column '{id_column}' not found in source dataframe.")

    expected = [str(col).strip() for col in expected_raw_cols if str(col).strip()]
    ids_df = pd.DataFrame({id_column: pd.Series(ids, dtype="string")})
    merged = ids_df.merge(df_year_t, on=id_column, how="left")
    raw = merged.drop(columns=[id_column], errors="ignore")

    numeric_set = set(NUMERIC_COLS)
    categorical_set = set(CATEGORICAL_COLS)
    datetime_set = set(DATETIME_COLS)
    n_rows = len(raw)

    for col in expected:
        if col in raw.columns:
            continue
        if col in numeric_set:
            raw[col] = pd.Series(np.nan, index=raw.index, dtype="Float64")
        elif col in datetime_set:
            raw[col] = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
        elif col in categorical_set:
            raw[col] = pd.Series(pd.NA, index=raw.index, dtype="string")
        else:
            # Conservative fallback for unknown raw columns.
            raw[col] = pd.Series(pd.NA, index=raw.index, dtype="string")

    if n_rows != len(ids_df):
        raise ValueError("Row alignment mismatch while building raw frame from ids.")

    return raw.loc[:, expected].copy()
