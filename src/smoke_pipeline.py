"""Smoke checks for raw->pipeline path with and without sklearn availability."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import (
    get_default_dataset_path,
    load_pede_workbook_with_metadata,
    make_temporal_pairs,
)
from src.pipeline_components import RawToModelFrameTransformer
from src.preprocessing import (
    _SKLEARN_AVAILABLE,
    build_preprocessing_bundle,
    build_pruning_plan_from_training_frame,
    get_expected_raw_feature_columns,
    get_feature_columns_for_model,
)


def _build_raw_from_ids(
    df_t: pd.DataFrame,
    ids: pd.Series,
    expected_raw_cols: list[str],
) -> pd.DataFrame:
    ids_df = pd.DataFrame({"RA": ids.astype("string")})
    raw_df = ids_df.merge(df_t, on="RA", how="left")
    return raw_df.loc[:, expected_raw_cols].copy()


def main() -> int:
    dataset_path = get_default_dataset_path()
    yearly_frames, _, _ = load_pede_workbook_with_metadata(dataset_path)

    _, y_train, ids_train = make_temporal_pairs(
        yearly_frames[2022],
        yearly_frames[2023],
        2022,
        2023,
    )
    if _SKLEARN_AVAILABLE:
        bundle = build_preprocessing_bundle(
            numeric_scaler="standard",
            enable_feature_engineering=False,
            enable_age_bucket=False,
        )
        expected_raw_cols = list(bundle["expected_raw_cols"])
        expected_model_cols = list(bundle["expected_model_cols"])
        X_raw_train = _build_raw_from_ids(
            yearly_frames[2022],
            ids_train,
            expected_raw_cols,
        )
        pruning_plan = build_pruning_plan_from_training_frame(
            X_train_raw=X_raw_train,
            enable_feature_engineering=False,
            enable_age_bucket=False,
        )
        from sklearn.linear_model import LogisticRegression

        from src.train_pipeline import build_model_pipeline

        pipeline = build_model_pipeline(
            model=LogisticRegression(max_iter=200),
            year_t=2022,
            scaler_strategy="standard",
            enable_feature_engineering=False,
            feature_pruning_plan=pruning_plan,
            strict_raw=True,
            enable_age_bucket=False,
        )
        pipeline.fit(X_raw_train, y_train)
        probs = pipeline.predict_proba(X_raw_train.iloc[:5])

        joblib = None
        try:
            import joblib as _joblib

            joblib = _joblib
        except ModuleNotFoundError:
            joblib = None

        if joblib is not None:
            artifacts_dir = Path("artifacts")
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            model_path = artifacts_dir / "smoke_pipeline.joblib"
            joblib.dump(pipeline, model_path)
            loaded_pipeline = joblib.load(model_path)
            loaded_probs = loaded_pipeline.predict_proba(X_raw_train.iloc[:5])
            print(
                "smoke_pipeline=ok mode=sklearn "
                f"rows_train={len(X_raw_train)} cols_raw={len(expected_raw_cols)} "
                f"cols_model={len(expected_model_cols)} probs_shape={probs.shape} "
                f"loaded_probs_shape={loaded_probs.shape} artifact={model_path.name}"
            )
        else:
            print(
                "smoke_pipeline=ok mode=sklearn_no_joblib "
                f"rows_train={len(X_raw_train)} cols_raw={len(expected_raw_cols)} "
                f"cols_model={len(expected_model_cols)} probs_shape={probs.shape}"
            )
        return 0

    expected_raw_cols = get_expected_raw_feature_columns()
    expected_model_cols = get_feature_columns_for_model()
    X_raw_train = _build_raw_from_ids(
        yearly_frames[2022],
        ids_train,
        expected_raw_cols,
    )
    pruning_plan = build_pruning_plan_from_training_frame(
        X_train_raw=X_raw_train,
        enable_feature_engineering=False,
        enable_age_bucket=False,
    )
    transformer = RawToModelFrameTransformer(
        year_t=2022,
        expected_raw_cols=expected_raw_cols,
        expected_model_cols=list(pruning_plan.get("kept_model_cols", expected_model_cols)),
        enable_feature_engineering=False,
        feature_pruning_plan=pruning_plan,
        strict_raw=True,
        enable_age_bucket=False,
    )
    X_model = transformer.transform(X_raw_train.iloc[:20].copy())
    print(
        "smoke_pipeline=ok mode=raw_to_model_only "
        f"rows_raw={20 if len(X_raw_train) >= 20 else len(X_raw_train)} "
        f"cols_raw={len(expected_raw_cols)} cols_model={X_model.shape[1]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
