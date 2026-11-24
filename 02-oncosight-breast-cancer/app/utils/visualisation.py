from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image


DISPLAY_IMAGE_SIZE = (512, 512)


def render_fixed_image(image, caption: str, size: tuple[int, int] = DISPLAY_IMAGE_SIZE) -> None:
    if isinstance(image, np.ndarray):
        display_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    else:
        display_image = image.convert("RGB")
    display_image = display_image.resize(size)
    st.image(display_image, caption=caption, width=size[0])


def probability_bar(label: str, value: float | None, help_text: str | None = None) -> None:
    if value is None:
        st.info(f"{label}: model not loaded or no input provided.")
        return
    clipped = float(np.clip(value, 0.0, 1.0))
    st.metric(label, f"{clipped:.1%}", help=help_text)
    st.progress(clipped)


def risk_category(score: float) -> tuple[str, str]:
    if score < 0.3:
        return "Low", "green"
    if score <= 0.6:
        return "Moderate", "orange"
    return "High", "red"


def render_risk_badge(category: str, color: str) -> None:
    st.markdown(
        f"""
        <div style="display:inline-block;padding:0.35rem 0.65rem;border-radius:6px;
        background:{color};color:white;font-weight:700;">{category} Risk</div>
        """,
        unsafe_allow_html=True,
    )


def render_interpretation_panel(title: str, probability: float | None, meaning: str, recommendation: str) -> None:
    if probability is None:
        st.info(f"{title}: no result available.")
        return

    category, color = risk_category(float(probability))
    color_map = {"green": "#2e7d32", "orange": "#ef6c00", "red": "#c62828"}
    display_color = color_map.get(color, "#455a64")

    st.markdown(
        f"""
        <div style="border:1px solid #d9dee3;border-left:8px solid {display_color};
        border-radius:10px;padding:0.9rem 1rem;margin:0.4rem 0 1rem 0;background:#fafbfc;">
            <div style="font-size:0.95rem;color:#5f6b76;margin-bottom:0.2rem;">{title}</div>
            <div style="font-size:1.6rem;font-weight:700;color:{display_color};margin-bottom:0.45rem;">
                {category} risk: {float(probability):.1%}
            </div>
            <div style="margin-bottom:0.45rem;"><strong>What this means:</strong> {meaning}</div>
            <div><strong>How to interpret it:</strong> {recommendation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def overlay_heatmap(original_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.38) -> np.ndarray:
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap = cv2.resize(heatmap.astype(np.float32), (original_rgb.shape[1], original_rgb.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_rgb.astype(np.uint8), 1 - alpha, colored, alpha, 0)
    return overlay


def feature_importance_chart(importances: pd.DataFrame | None) -> None:
    if importances is None or importances.empty:
        st.info("Feature importance is unavailable for the loaded clinical model.")
        return
    fig = px.bar(
        importances.sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        orientation="h",
        labels={"importance": "Importance", "feature": "Feature"},
        height=360,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
