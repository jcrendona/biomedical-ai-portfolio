from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

from modules.fusion import weighted_late_fusion
from modules.pcam_module import predict_metastasis_probability, render_pcam_status
from modules.rsna_module import generate_gradcam, predict_cancer_probability, render_rsna_status
from modules.seer_module import predict_seer_risk, render_seer_status, top_feature_importance
from utils.preprocessing import load_uploaded_image
from utils.visualisation import (
    feature_importance_chart,
    probability_bar,
    render_fixed_image,
    render_interpretation_panel,
    render_risk_badge,
    risk_category,
)


st.set_page_config(page_title="OncoSight: Breast Cancer Decision-Support Prototype", layout="wide")

st.title("OncoSight: Multimodal Breast Cancer Risk Triage Support")
st.caption("Academic prototype for risk explanation, case prioritization, and clinical decision support")
st.image("https://upload.wikimedia.org/wikipedia/en/0/07/Breast_Cancer_Pink_Ribbon_Transparent.png", caption="Breast Cancer")


DEFAULT_SEER_VALUES = {
    "age": 50,
    "race": "White",
    "marital_status": "Married",
    "t_stage": "T1",
    "n_stage": "N1",
    "sixth_stage": "IIA",
    "grade": "Moderately differentiated",
    "tumor_size": 1,
    "estrogen_status": "Positive",
    "progesterone_status": "Positive",
    "regional_node_examined": 12,
    "regional_node_positive": 1,
}

DEFAULT_FUSION_WEIGHTS = {"mammography": 0.50, "seer": 0.30, "pcam": 0.20}
WEIGHT_KEYS = {
    "mammography": "weight_mammography",
    "seer": "weight_seer",
    "pcam": "weight_pcam",
}
TOGGLE_KEYS = {
    "mammography": "toggle_mammography",
    "seer": "toggle_seer",
    "pcam": "toggle_pcam",
}
MODULE_COPY = {
    "mammography": {
        "sidebar_label": "1. Initial Mammography Review",
        "sidebar_help": "Primary imaging screen to flag suspicious mammographic patterns for triage.",
        "weight_label": "Mammography",
        "tab_label": "1. Mammography",
        "result_label": "Mammography Triage Signal",
        "probability_label": "Estimated probability of suspicious cancer-related imaging features",
        "help_text": "Mammography model estimate. This is not a definitive cancer diagnosis.",
        "meanings": {
            "Low": (
                "The mammography-based system identified limited unusual tissue patterns associated with malignancy in this image.",
                "This may support routine follow-up if the imaging review and clinical context remain reassuring.",
            ),
            "Moderate": (
                "The mammography-based system identified an intermediate level of unusual tissue patterns that warrant closer review.",
                "This result should be correlated with radiologist interpretation, BI-RADS assessment, symptoms, and prior studies.",
            ),
            "High": (
                "The mammography-based system identified a high level of unusual tissue patterns associated with possible malignancy.",
                "This case should be prioritized for prompt clinical review and confirmatory diagnostic workup; it is not a final diagnosis on its own.",
            ),
        },
    },
    "seer": {
        "sidebar_label": "2. Clinical Context",
        "sidebar_help": "Clinical and pathology-related variables used to estimate prognostic triage risk.",
        "weight_label": "Clinical Context",
        "tab_label": "2. Clinical Variables",
        "result_label": "Clinical Triage Signal",
        "probability_label": "Estimated probability of elevated clinical prognostic risk",
        "help_text": "Clinical prognostic estimate based on structured variables. It does not replace clinical judgment.",
        "meanings": {
            "Low": (
                "The clinical risk model identified a relatively lower-risk prognostic profile based on the entered variables.",
                "This may indicate a more favorable triage context, but it should still be interpreted with stage, treatment plan, and real-world course.",
            ),
            "Moderate": (
                "The clinical risk model identified a mixed or intermediate-risk prognostic profile.",
                "This should be reviewed alongside imaging findings, pathology, and the treatment plan before assigning case priority.",
            ),
            "High": (
                "The clinical risk model identified a higher-alert prognostic profile from the entered variables.",
                "This supports prioritizing multidisciplinary review and closer attention to the patient pathway, but it is not a final cancer diagnosis.",
            ),
        },
    },
    "pcam": {
        "sidebar_label": "3. Complementary Histopathology (PoC)",
        "sidebar_help": "Experimental histopathology patch review to flag patterns compatible with nodal metastasis.",
        "weight_label": "Histopathology",
        "tab_label": "3. Histopathology",
        "result_label": "Histopathology Triage Signal",
        "probability_label": "Estimated probability of patterns compatible with metastasis",
        "help_text": "Exploratory PCam output. This is a proof-of-concept module, not a clinically validated integrated diagnosis.",
        "meanings": {
            "Low": (
                "The histopathology-based system identified limited patch-level patterns associated with metastasis.",
                "This may be somewhat reassuring for the specific patch, but it does not replace full pathology review.",
            ),
            "Moderate": (
                "The histopathology-based system identified an intermediate level of suspicious patch-level features.",
                "This should be interpreted cautiously and correlated with the full slide or specimen rather than in isolation.",
            ),
            "High": (
                "The histopathology-based system identified a high level of patch-level features associated with possible metastasis.",
                "This should be treated as a technical alert for expert pathology review, not as definitive confirmation.",
            ),
        },
    },
}


def _sync_weight_slider(toggle_key: str, slider_key: str, default: float, force_full_weight: bool = False) -> None:
    if f"{slider_key}_saved" not in st.session_state:
        st.session_state[f"{slider_key}_saved"] = default
    if slider_key not in st.session_state:
        st.session_state[slider_key] = default

    if force_full_weight and st.session_state.get(toggle_key, True):
        current = st.session_state.get(slider_key, default)
        if current < 1.0:
            st.session_state[f"{slider_key}_saved"] = current if current > 0 else st.session_state.get(
                f"{slider_key}_saved", default
            )
        st.session_state[slider_key] = 1.0
    elif st.session_state.get(toggle_key, True):
        if st.session_state.get(slider_key, 0.0) == 0.0:
            restored = st.session_state.get(f"{slider_key}_saved", default)
            st.session_state[slider_key] = restored if restored > 0 else default
    else:
        current = st.session_state.get(slider_key, default)
        if current > 0:
            st.session_state[f"{slider_key}_saved"] = current
        st.session_state[slider_key] = 0.0


def _rebalance_weight_sliders(changed_key: str | None = None) -> None:
    active_modalities = {
        name: st.session_state.get(toggle_key, True) for name, toggle_key in TOGGLE_KEYS.items()
    }
    active_names = [name for name, enabled in active_modalities.items() if enabled]
    independent_mode = st.session_state.get("independent_weight_mode", False)

    for name, slider_key in WEIGHT_KEYS.items():
        if not active_modalities[name]:
            st.session_state[slider_key] = 0.0

    if not active_names:
        return

    if len(active_names) == 1:
        st.session_state[WEIGHT_KEYS[active_names[0]]] = 1.0
        return

    if independent_mode:
        return

    active_keys = [WEIGHT_KEYS[name] for name in active_names]
    if changed_key in active_keys:
        changed_value = min(max(float(st.session_state.get(changed_key, 0.0)), 0.0), 1.0)
        st.session_state[changed_key] = changed_value
        other_keys = [key for key in active_keys if key != changed_key]
        remaining = max(0.0, 1.0 - changed_value)
        other_total = sum(float(st.session_state.get(key, 0.0)) for key in other_keys)

        if other_keys:
            if other_total > 0:
                for key in other_keys:
                    st.session_state[key] = float(st.session_state.get(key, 0.0)) / other_total * remaining
            else:
                defaults_total = sum(
                    DEFAULT_FUSION_WEIGHTS[name] for name, key in WEIGHT_KEYS.items() if key in other_keys
                )
                for name, key in WEIGHT_KEYS.items():
                    if key in other_keys:
                        st.session_state[key] = (
                            DEFAULT_FUSION_WEIGHTS[name] / defaults_total * remaining if defaults_total > 0 else 0.0
                        )
        return

    total = sum(float(st.session_state.get(key, 0.0)) for key in active_keys)
    if total <= 0:
        defaults_total = sum(DEFAULT_FUSION_WEIGHTS[name] for name in active_names)
        for name in active_names:
            st.session_state[WEIGHT_KEYS[name]] = DEFAULT_FUSION_WEIGHTS[name] / defaults_total
        return

    for key in active_keys:
        st.session_state[key] = float(st.session_state.get(key, 0.0)) / total


def _handle_weight_toggle_change() -> None:
    for name, slider_key in WEIGHT_KEYS.items():
        _sync_weight_slider(
            TOGGLE_KEYS[name],
            slider_key,
            DEFAULT_FUSION_WEIGHTS[name],
            force_full_weight=False,
        )
    _rebalance_weight_sliders()


def _handle_weight_slider_change(slider_key: str) -> None:
    _rebalance_weight_sliders(changed_key=slider_key)


def _handle_independent_mode_change() -> None:
    _rebalance_weight_sliders()


def _module_summary_note(module_key: str, score: float, has_roi: bool = False) -> str:
    percent = f"{score:.1%}"
    if module_key == "mammography":
        roi_sentence = (
            " The ROI is highlighted in the Grad-CAM heatmap to show the image regions that contributed most strongly to the model output."
            if has_roi
            else " ROI visualization is unavailable for the currently loaded architecture."
        )
        return (
            f"The mammography-based system identified unusual tissue patterns with an estimated {percent} probability of being cancer-related."
            f"{roi_sentence} This output is intended for triage and case prioritization, not as a final diagnosis of cancer."
        )
    if module_key == "seer":
        return (
            f"The clinical risk model estimated a {percent} probability of elevated prognostic risk based on the entered patient variables."
            " This result should help prioritize review intensity and follow-up planning, not replace clinician judgment or establish a final diagnosis."
        )
    return (
        f"The histopathology-based system identified patch-level tissue patterns with an estimated {percent} probability of being compatible with metastasis."
        " This proof-of-concept output is intended to support triage and prioritization of expert review, not to confirm metastatic disease on its own."
    )


def _final_triage_note(score: float) -> tuple[str, str]:
    category, _ = risk_category(score)
    if category == "Low":
        return (
            "Lower triage priority",
            "The integrated risk profile suggests that this patient may require routine follow-up rather than urgent escalation, assuming the broader clinical picture is consistent.",
        )
    if category == "Moderate":
        return (
            "Priority clinical review recommended",
            "The integrated risk profile suggests that this patient should receive timely clinical review and correlation across imaging, pathology, and clinical history.",
        )
    return (
        "Urgent attention recommended",
        "The integrated risk profile suggests that this patient should be prioritized for urgent clinical attention and confirmatory diagnostic review.",
    )


def _render_module_result(module_key: str, score: float | None) -> None:
    copy = MODULE_COPY[module_key]
    probability_bar(copy["probability_label"], score, copy["help_text"])
    if score is None:
        return
    category, _ = risk_category(score)
    meaning, recommendation = copy["meanings"][category]
    render_interpretation_panel(copy["result_label"], score, meaning, recommendation)

with st.sidebar:
    st.header("Information Sources")
    st.caption("Enable modules in a triage-oriented clinical reading order.")
    use_rsna = st.toggle(
        MODULE_COPY["mammography"]["sidebar_label"],
        value=True,
        key="toggle_mammography",
        help=MODULE_COPY["mammography"]["sidebar_help"],
        on_change=_handle_weight_toggle_change,
    )
    use_pcam = st.toggle(
        MODULE_COPY["pcam"]["sidebar_label"],
        value=True,
        key="toggle_pcam",
        help=MODULE_COPY["pcam"]["sidebar_help"],
        on_change=_handle_weight_toggle_change,
    )
    use_seer = st.toggle(
        MODULE_COPY["seer"]["sidebar_label"],
        value=True,
        key="toggle_seer",
        help=MODULE_COPY["seer"]["sidebar_help"],
        on_change=_handle_weight_toggle_change,
    )
    active_count = sum([use_rsna, use_pcam, use_seer])

    st.header("Source Weights")
    weight_help = (
        "Mammography starts at 50% because it is the primary screening entry point. "
        "These weights are prototype heuristics and are not clinically validated."
    )
    independent_weight_mode = st.toggle(
        "Allow independent weights",
        value=False,
        key="independent_weight_mode",
        help="Allow enabled sources to keep arbitrary values instead of forcing the total to sum to 1.",
        on_change=_handle_independent_mode_change,
    )
    _sync_weight_slider(
        "toggle_mammography",
        "weight_mammography",
        DEFAULT_FUSION_WEIGHTS["mammography"],
        force_full_weight=use_rsna and active_count == 1,
    )
    _sync_weight_slider(
        "toggle_seer",
        "weight_seer",
        DEFAULT_FUSION_WEIGHTS["seer"],
        force_full_weight=use_seer and active_count == 1,
    )
    _sync_weight_slider(
        "toggle_pcam",
        "weight_pcam",
        DEFAULT_FUSION_WEIGHTS["pcam"],
        force_full_weight=use_pcam and active_count == 1,
    )
    if not independent_weight_mode and active_count > 1:
        _rebalance_weight_sliders()

    w1 = st.slider(
        MODULE_COPY["mammography"]["weight_label"],
        0.0,
        1.0,
        DEFAULT_FUSION_WEIGHTS["mammography"],
        0.05,
        key="weight_mammography",
        disabled=not use_rsna or active_count == 1,
        help=weight_help,
        on_change=_handle_weight_slider_change,
        args=("weight_mammography",),
    )
    w2 = st.slider(
        MODULE_COPY["seer"]["weight_label"],
        0.0,
        1.0,
        DEFAULT_FUSION_WEIGHTS["seer"],
        0.05,
        key="weight_seer",
        disabled=not use_seer or active_count == 1,
        on_change=_handle_weight_slider_change,
        args=("weight_seer",),
    )
    w3 = st.slider(
        MODULE_COPY["pcam"]["weight_label"],
        0.0,
        1.0,
        DEFAULT_FUSION_WEIGHTS["pcam"],
        0.05,
        key="weight_pcam",
        disabled=not use_pcam or active_count == 1,
        on_change=_handle_weight_slider_change,
        args=("weight_pcam",),
    )

    raw_weights = {"mammography": w1, "seer": w2, "pcam": w3}
    active_modalities = {
        "mammography": use_rsna,
        "seer": use_seer,
        "pcam": use_pcam,
    }
    active_weight_total = sum(raw_weights[name] for name, enabled in active_modalities.items() if enabled)
    if active_count == 0:
        st.warning("Enable at least one source to compute the integrated triage score.")
        weights = {name: 0.0 for name in raw_weights}
    elif active_weight_total <= 0:
        st.warning("At least one enabled source must have a weight above zero. Default weights will be used.")
        fallback_total = sum(
            DEFAULT_FUSION_WEIGHTS[name] for name, enabled in active_modalities.items() if enabled
        )
        weights = {
            name: (
                DEFAULT_FUSION_WEIGHTS[name] / fallback_total
                if active_modalities[name] and fallback_total > 0
                else 0.0
            )
            for name in DEFAULT_FUSION_WEIGHTS
        }
    elif active_count == 1:
        weights = {name: 1.0 if active_modalities[name] else 0.0 for name in raw_weights}
    else:
        weights = {
            name: (raw_weights[name] / active_weight_total if active_modalities[name] else 0.0)
            for name in raw_weights
        }

    if active_count < 3:
        st.info(
            "Effective weights: "
            f"Mammography={weights['mammography']:.2f}, "
            f"Clinical={weights['seer']:.2f}, "
            f"Histopathology={weights['pcam']:.2f}"
        )


scores = {"mammography": None, "seer": None, "pcam": None}
gradcam_overlay = None
seer_importance = None

tab_rsna, tab_seer, tab_pcam = st.tabs(
    [
        MODULE_COPY["mammography"]["tab_label"],
        MODULE_COPY["seer"]["tab_label"],
        MODULE_COPY["pcam"]["tab_label"],
    ]
)

with tab_rsna:
    st.subheader("Mammography Review")
    if not use_rsna:
        st.info("Mammography module disabled.")
    else:
        rsna_model, rsna_framework, _ = render_rsna_status()
        mammo_file = st.file_uploader("Upload mammography image", type=["jpg", "jpeg", "png", "dcm"], key="mammo")
        if mammo_file:
            try:
                mammo_image = load_uploaded_image(mammo_file)
                render_fixed_image(mammo_image, "Uploaded mammography image")
                if rsna_model is not None:
                    scores["mammography"] = predict_cancer_probability(rsna_model, rsna_framework, mammo_image)
                    gradcam_overlay = generate_gradcam(rsna_model, rsna_framework, mammo_image)
                    _render_module_result("mammography", scores["mammography"])
                    st.caption(_module_summary_note("mammography", scores["mammography"], has_roi=gradcam_overlay is not None))
                    if gradcam_overlay is not None:
                        render_fixed_image(gradcam_overlay, "Grad-CAM heatmap")
                    else:
                        st.info("Grad-CAM is unavailable for the loaded architecture.")
                else:
                    st.info("Model not loaded. Image upload is available, but inference is disabled.")
            except Exception as exc:
                st.error(f"Mammography input could not be processed: {exc}")

with tab_seer:
    st.subheader("Clinical Context and Prognostic Review")
    if not use_seer:
        st.info("Clinical module disabled.")
    else:
        seer_model, seer_preprocessor, _ = render_seer_status()
        with st.form("seer_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Age", 18, 100, DEFAULT_SEER_VALUES["age"])
                race = st.selectbox("Race", ["White", "Black", "Other"])
                marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed", "Separated"])
                t_stage = st.selectbox("T Stage", ["T1", "T2", "T3", "T4"])
            with c2:
                n_stage = st.selectbox("N Stage", ["N1", "N2", "N3"])
                sixth_stage = st.selectbox("6th Stage", ["IIA", "IIB", "IIIA", "IIIB", "IIIC"])
                grade = st.selectbox(
                    "Grade/Differentiate",
                    ["Well differentiated", "Moderately differentiated", "Poorly differentiated", "Undifferentiated"],
                )
                tumor_size = st.number_input("Tumor Size (mm)", 1, 200, DEFAULT_SEER_VALUES["tumor_size"])
            with c3:
                estrogen_status = st.selectbox("Estrogen Status", ["Positive", "Negative"])
                progesterone_status = st.selectbox("Progesterone Status", ["Positive", "Negative"])
                regional_node_examined = st.number_input(
                    "Regional Node Examined", 0, 100, DEFAULT_SEER_VALUES["regional_node_examined"]
                )
                regional_node_positive = st.number_input(
                    "Regional Node Positive", 0, 100, DEFAULT_SEER_VALUES["regional_node_positive"]
                )

            values = {
                "age": age,
                "race": race,
                "marital_status": marital_status,
                "t_stage": t_stage,
                "n_stage": n_stage,
                "sixth_stage": sixth_stage,
                "grade": grade,
                "tumor_size": tumor_size,
                "estrogen_status": estrogen_status,
                "progesterone_status": progesterone_status,
                "regional_node_examined": regional_node_examined,
                "regional_node_positive": regional_node_positive,
            }
            seer_submitted = st.form_submit_button("Update clinical risk estimate")

        has_changed_from_defaults = values != DEFAULT_SEER_VALUES
        last_seer_values = st.session_state.get("seer_last_values")
        has_pending_changes = last_seer_values is not None and values != last_seer_values

        if seer_submitted:
            if not has_changed_from_defaults:
                st.session_state["seer_score"] = None
                st.session_state["seer_warning"] = "Modify at least one clinical field before requesting a clinical risk estimate."
                st.session_state["seer_importance"] = None
                st.session_state["seer_last_values"] = None
            elif seer_model is not None:
                score, seer_warning = predict_seer_risk(seer_model, seer_preprocessor, values)
                st.session_state["seer_score"] = score
                st.session_state["seer_warning"] = seer_warning
                st.session_state["seer_importance"] = top_feature_importance(seer_model, seer_preprocessor)
                st.session_state["seer_last_values"] = values.copy()
            else:
                st.session_state["seer_score"] = None
                st.session_state["seer_warning"] = "Model not loaded. The clinical form is available, but inference is disabled."
                st.session_state["seer_importance"] = None
                st.session_state["seer_last_values"] = None

        if not has_changed_from_defaults and st.session_state.get("seer_last_values") is None:
            st.info("Adjust at least one clinical variable and click `Update clinical risk estimate` to generate a result.")
        elif has_pending_changes:
            st.info("Clinical inputs changed. Click `Update clinical risk estimate` to refresh the result.")

        scores["seer"] = st.session_state.get("seer_score")
        seer_warning = st.session_state.get("seer_warning")
        if seer_warning:
            st.warning(seer_warning)
        if scores["seer"] is not None:
            _render_module_result("seer", scores["seer"])
            st.caption(_module_summary_note("seer", scores["seer"]))
            seer_importance = st.session_state.get("seer_importance")
            feature_importance_chart(seer_importance)

with tab_pcam:
    st.subheader("Complementary Histopathology Review")
    if not use_pcam:
        st.info("Histopathology module disabled.")
    else:
        pcam_model, pcam_framework, _ = render_pcam_status()
        pcam_file = st.file_uploader("Upload histopathology patch", type=["jpg", "jpeg", "png"], key="pcam")
        if pcam_file:
            try:
                pcam_image = load_uploaded_image(pcam_file)
                render_fixed_image(pcam_image, "Uploaded histopathology patch")
                scores["pcam"] = predict_metastasis_probability(pcam_model, pcam_framework, pcam_image)
                if scores["pcam"] is None:
                    st.info("Model not loaded for executable inference. A prototype placeholder example will be shown.")
                    scores["pcam"] = 0.42
                _render_module_result("pcam", scores["pcam"])
                st.caption(_module_summary_note("pcam", scores["pcam"]))
            except Exception as exc:
                st.error(f"Histopathology input could not be processed: {exc}")
        elif pcam_model is None:
            st.info("Model not loaded. Prototype example output: 42%.")

st.divider()
st.subheader("Integrated Triage Summary")
st.caption(
    "These probabilities are model-based estimates for clinical decision support and patient prioritization. "
    "They must be interpreted with the full clinical context and do not represent a final diagnosis of cancer."
)

enabled_scores = {
    "mammography": scores["mammography"] if use_rsna else None,
    "seer": scores["seer"] if use_seer else None,
    "pcam": scores["pcam"] if use_pcam else None,
}
fusion = weighted_late_fusion(
    enabled_scores["mammography"],
    enabled_scores["seer"],
    enabled_scores["pcam"],
    weights,
)

c1, c2, c3 = st.columns(3)
with c1:
    probability_bar(MODULE_COPY["mammography"]["weight_label"], enabled_scores["mammography"], MODULE_COPY["mammography"]["help_text"])
with c2:
    probability_bar(MODULE_COPY["seer"]["weight_label"], enabled_scores["seer"], MODULE_COPY["seer"]["help_text"])
with c3:
    probability_bar(MODULE_COPY["pcam"]["weight_label"], enabled_scores["pcam"], MODULE_COPY["pcam"]["help_text"])

if fusion.score is not None:
    category, color = risk_category(fusion.score)
    st.metric("Final Integrated Risk Factor", f"{fusion.score:.1%}")
    st.progress(float(fusion.score))
    render_risk_badge(category, color)
    final_meaning = {
        "Low": (
            "The available modules together suggest a lower-alert triage profile in this prototype.",
            "This case may require standard follow-up rather than urgent escalation if the broader clinical picture is consistent.",
        ),
        "Moderate": (
            "The available modules together suggest an intermediate-alert triage profile.",
            "This case should receive timely clinical review with correlation across imaging, pathology, and patient history.",
        ),
        "High": (
            "The available modules together suggest a high-alert triage profile.",
            "This case should be prioritized for urgent clinical attention and confirmatory diagnostic review.",
        ),
    }
    meaning, recommendation = final_meaning[category]
    render_interpretation_panel("Integrated Risk Interpretation", fusion.score, meaning, recommendation)
    triage_title, triage_message = _final_triage_note(fusion.score)
    st.warning(
        f"{triage_title}: {triage_message} This integrated score is intended for triage and decision support only, not as a final diagnosis of cancer."
    )
    st.caption(
        "Effective weights used for integration: "
        + ", ".join(f"{MODULE_COPY[name]['weight_label']}={weight:.2f}" for name, weight in fusion.effective_weights.items())
    )
else:
    st.info("No results are available for integration yet. Upload inputs or enable a module with an available output.")

st.error(
    "This prototype is a triage and clinical decision-support tool for academic demonstration only. "
    "It has not been clinically validated and must not be used as a final cancer diagnosis or as the sole basis for patient-care decisions."
)
