from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from utils.preprocessing import (
    app_root,
    fallback_seer_numeric_frame,
    project_root,
    seer_form_to_frame,
    typical_seer_reference,
)


def _seer_asset_dirs() -> list[Path]:
    return [app_root() / "models" / "seer", project_root() / "models" / "seer"]


def _find_seer_asset(names: list[str]) -> Path | None:
    for folder in _seer_asset_dirs():
        if not folder.exists():
            continue
        for name in names:
            path = folder / name
            if path.exists():
                return path
    return None


def _enable_legacy_column_transformer_pickle_support() -> None:
    # scikit-learn 1.6 serialized ColumnTransformer remainder columns using an
    # internal helper that is no longer present in newer releases.
    try:
        import sklearn.compose._column_transformer as column_transformer

        if not hasattr(column_transformer, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            column_transformer._RemainderColsList = _RemainderColsList
    except Exception:
        pass


@st.cache_resource(show_spinner=False)
def load_seer_assets() -> tuple[Any | None, Any | None, str | None, str | None, str | None]:
    try:
        model_path = _find_seer_asset(
            [
                "best_seer_model.joblib",
                "random_forest_balanced.joblib",
                "hist_gradient_boosting.joblib",
                "logistic_regression_balanced.joblib",
                "dummy_prior.joblib",
            ]
        )
        preprocessor_path = _find_seer_asset(["seer_preprocessor.joblib"])

        model = None
        preprocessor = None
        warnings: list[str] = []
        import joblib

        if model_path:
            try:
                model = joblib.load(model_path)
            except Exception as exc:
                return None, None, None, None, f"SEER model could not be loaded: {exc}"

        if preprocessor_path:
            try:
                _enable_legacy_column_transformer_pickle_support()
                preprocessor = joblib.load(preprocessor_path)
            except Exception as exc:
                warnings.append(
                    "SEER preprocessor could not be loaded, likely due to a scikit-learn version mismatch. "
                    f"Fallback scaling will be used instead. Details: {exc}"
                )

        if not model:
            return None, preprocessor, None, str(preprocessor_path) if preprocessor_path else None, (
                "No SEER model found. Place a Random Forest .pkl/.joblib model in app/models."
            )
        if preprocessor is None and preprocessor_path and not warnings:
            warnings.append("SEER preprocessor was not loaded; fallback scaling will be used.")
        if preprocessor is None and not preprocessor_path:
            warnings.append("No separate SEER preprocessor/scaler found; fallback scaling will be used.")

        return (
            model,
            preprocessor,
            str(model_path),
            str(preprocessor_path) if preprocessor_path else None,
            " ".join(warnings) if warnings else None,
        )
    except Exception as exc:
        return None, None, None, None, f"SEER asset loading failed: {exc}"


def _positive_probability(model: Any, X: Any) -> float:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        if prob.shape[1] == 1:
            return float(prob[0, 0])
        return float(prob[0, 1])
    if hasattr(model, "decision_function"):
        score = float(np.ravel(model.decision_function(X))[0])
        return float(1.0 / (1.0 + np.exp(-score)))
    pred = model.predict(X)
    return float(np.clip(np.ravel(pred)[0], 0.0, 1.0))


def _align_seer_input(df: pd.DataFrame, preprocessor: Any) -> tuple[pd.DataFrame, str | None]:
    if not hasattr(preprocessor, "feature_names_in_"):
        return df, None

    aligned = df.copy()
    defaults = {
        "differentiate": aligned["Grade"].iloc[0] if "Grade" in aligned else "Moderately differentiated",
        "A Stage": "Regional",
        "Survival Months": 60,
    }
    used_defaults: list[str] = []

    for column in preprocessor.feature_names_in_:
        if column not in aligned.columns:
            aligned[column] = defaults.get(column, 0)
            used_defaults.append(column)

    if used_defaults:
        return (
            aligned,
            "SEER input was auto-completed for missing training-era fields: "
            + ", ".join(used_defaults)
            + ". These defaults are heuristic and may affect the score.",
        )
    return aligned, None


def predict_seer_risk(model: Any, preprocessor: Any | None, values: dict[str, Any]) -> tuple[float | None, str | None]:
    df = seer_form_to_frame(values)
    try:
        if preprocessor is not None:
            df, alignment_warning = _align_seer_input(df, preprocessor)
            X = preprocessor.transform(df)
            return float(np.clip(_positive_probability(model, X), 0.0, 1.0)), alignment_warning

        from sklearn.preprocessing import StandardScaler

        numeric = fallback_seer_numeric_frame(df)
        ref = typical_seer_reference()
        scaler = StandardScaler().fit(ref[numeric.columns])
        X = scaler.transform(numeric)
        return float(np.clip(_positive_probability(model, X), 0.0, 1.0)), (
            "Fallback StandardScaler fitted on typical SEER ranges was used because no saved scaler/preprocessor was loaded."
        )
    except Exception as exc:
        return None, f"SEER inference failed: {exc}"


def top_feature_importance(model: Any, preprocessor: Any | None, n: int = 10) -> pd.DataFrame | None:
    try:
        estimator = model
        if hasattr(model, "named_steps"):
            estimator = list(model.named_steps.values())[-1]
        if not hasattr(estimator, "feature_importances_"):
            return None

        names = None
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            names = list(preprocessor.get_feature_names_out())
        if names is None:
            names = [f"feature_{i}" for i in range(len(estimator.feature_importances_))]

        df = pd.DataFrame({"feature": names[: len(estimator.feature_importances_)], "importance": estimator.feature_importances_})
        return df.sort_values("importance", ascending=False).head(n)
    except Exception:
        return None


def render_seer_status() -> tuple[Any | None, Any | None, str | None]:
    model, preprocessor, model_path, preprocessor_path, warning = load_seer_assets()
    if model_path:
        st.caption(f"SEER model artifact: `{Path(model_path).name}`")
    if preprocessor_path:
        st.caption(f"SEER preprocessor artifact: `{Path(preprocessor_path).name}`")
    if warning:
        st.warning(warning)
    return model, preprocessor, warning
