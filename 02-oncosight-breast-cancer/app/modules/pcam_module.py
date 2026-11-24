from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from utils.preprocessing import find_model_file, find_saved_model_dir, preprocess_image


PCAM_DISCLAIMER = (
    "Proof-of-concept histopathology module for lymph node patch classification. "
    "It is not clinically integrated with mammography because it comes from unmatched cohorts."
)


def _build_small_patch_cnn():
    import torch.nn as nn

    class SmallPatchCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(128, 1)

        def forward(self, x):
            return self.head(self.net(x).flatten(1)).squeeze(1)

    return SmallPatchCNN()


@st.cache_resource(show_spinner=False)
def load_pcam_model() -> tuple[Any | None, str | None, str | None, str | None]:
    try:
        keras_path = find_model_file(["pcam", "histopath", "cnn"], (".keras", ".h5"))
        saved_model = find_saved_model_dir(["pcam", "histopath", "cnn"])
        if keras_path or saved_model:
            import tensorflow as tf

            path = str(keras_path or saved_model)
            return tf.keras.models.load_model(path), "tensorflow", path, None

        torch_path = find_model_file(["pcam", "cnn", "small"], (".pt", ".pth"))
        if torch_path:
            try:
                import torch

                checkpoint = torch.load(torch_path, map_location="cpu")
                model = _build_small_patch_cnn()
                state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                return model, "pytorch", str(torch_path), None
            except Exception as exc:
                return None, None, str(torch_path), f"PCam PyTorch checkpoint could not be loaded: {exc}"
        return None, None, None, "No PCam model found. The interface will show a prototype placeholder output."
    except Exception as exc:
        return None, None, None, f"PCam model loading failed: {exc}"


def predict_metastasis_probability(model: Any, framework: str | None, image) -> float | None:
    if framework == "tensorflow":
        arr = preprocess_image(image, (96, 96))
        pred = model.predict(arr, verbose=0)
        return float(np.clip(np.ravel(pred)[0], 0.0, 1.0))
    if framework == "pytorch":
        import torch
        from torchvision import transforms

        transform = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor()])
        x = transform(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).detach().cpu().numpy().ravel()[0]
        return float(np.clip(prob, 0.0, 1.0))
    return None


def render_pcam_status() -> tuple[Any | None, str | None, str | None]:
    st.warning(PCAM_DISCLAIMER)
    model, framework, path, error = load_pcam_model()
    if path:
        st.caption(f"PCam model artifact: `{Path(path).name}`")
    if error:
        st.info(error)
    return model, framework, error
