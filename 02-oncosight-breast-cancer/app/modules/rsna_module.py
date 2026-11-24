from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from utils.preprocessing import find_model_file, find_saved_model_dir, image_to_uint8, preprocess_image
from utils.visualisation import overlay_heatmap


@st.cache_resource(show_spinner=False)
def load_rsna_model() -> tuple[Any | None, str | None, str | None, str | None]:
    try:
        torch_path = find_model_file(["rsna", "resnet"], (".pt", ".pth"))
        if torch_path:
            try:
                import torch
                import torch.nn as nn
                from torchvision.models import ResNet50_Weights, resnet50

                checkpoint = torch.load(torch_path, map_location="cpu")
                model = resnet50(weights=None)
                model.fc = nn.Linear(model.fc.in_features, 1)

                state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                return model, "pytorch", str(torch_path), None
            except Exception as exc:
                return None, "pytorch", str(torch_path), f"RSNA PyTorch checkpoint found but could not be loaded: {exc}"

        keras_path = find_model_file(["rsna", "resnet", "mammography"], (".keras", ".h5"))
        saved_model = find_saved_model_dir(["rsna", "resnet", "mammography"])
        if keras_path or saved_model:
            import tensorflow as tf

            path = str(keras_path or saved_model)
            return tf.keras.models.load_model(path), "tensorflow", path, None

        return None, None, None, "No RSNA model found. Place a .pt/.pth, .keras, .h5, or SavedModel artifact in app/models."
    except Exception as exc:
        return None, None, None, f"RSNA model loading failed: {exc}"


def predict_cancer_probability(model: Any, framework: str | None, image) -> float:
    if framework == "pytorch":
        import torch
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        x = transform(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).detach().cpu().numpy().ravel()[0]
        return float(np.clip(prob, 0.0, 1.0))

    arr = preprocess_image(image, (224, 224))
    pred = model.predict(arr, verbose=0)
    prob = float(np.ravel(pred)[0])
    return float(np.clip(prob, 0.0, 1.0))


def _last_conv_layer_name(model: Any) -> str | None:
    try:
        import tensorflow as tf

        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
            if hasattr(layer, "layers"):
                for sub in reversed(layer.layers):
                    if isinstance(sub, tf.keras.layers.Conv2D):
                        return sub.name
    except Exception:
        return None
    return None


def generate_gradcam(model: Any, framework: str | None, image) -> np.ndarray | None:
    if framework == "pytorch":
        try:
            import torch
            from torchvision import transforms

            activations = []
            gradients = []

            def forward_hook(_module, _inp, out):
                activations.append(out.detach())

            def backward_hook(_module, _grad_in, grad_out):
                gradients.append(grad_out[0].detach())

            target_layer = model.layer4[-1]
            h1 = target_layer.register_forward_hook(forward_hook)
            h2 = target_layer.register_full_backward_hook(backward_hook)

            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            x = transform(image.convert("RGB")).unsqueeze(0)
            model.zero_grad(set_to_none=True)
            logits = model(x)
            score = logits[:, 0].sum()
            score.backward()

            h1.remove()
            h2.remove()

            if not activations or not gradients:
                return None
            acts = activations[0][0]
            grads = gradients[0][0]
            weights = grads.mean(dim=(1, 2))
            heatmap = torch.sum(weights[:, None, None] * acts, dim=0)
            heatmap = torch.relu(heatmap).detach().cpu().numpy()
            original = image_to_uint8(image)
            return overlay_heatmap(original, heatmap)
        except Exception:
            return None

    try:
        import tensorflow as tf

        arr = preprocess_image(image, (224, 224))
        last_conv = _last_conv_layer_name(model)
        if not last_conv:
            return None

        conv_layer = None
        try:
            conv_layer = model.get_layer(last_conv)
            grad_model = tf.keras.models.Model(model.inputs, [conv_layer.output, model.output])
        except Exception:
            for layer in model.layers:
                if hasattr(layer, "get_layer"):
                    try:
                        conv_layer = layer.get_layer(last_conv)
                        backbone = tf.keras.models.Model(layer.inputs, [conv_layer.output, layer.output])
                        break
                    except Exception:
                        continue
            else:
                return None
            grad_model = tf.keras.models.Model(model.inputs, [backbone(model.input)[0], model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(arr)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        original = image_to_uint8(image)
        return overlay_heatmap(original, heatmap)
    except Exception:
        return None


def render_rsna_status() -> tuple[Any | None, str | None, str | None]:
    model, framework, path, error = load_rsna_model()
    if path:
        st.caption(f"RSNA model artifact: `{Path(path).name}` ({framework})")
    if error:
        st.warning(error)
    st.info(
        "For this module, recall/sensitivity for the cancer class is a key screening metric because the RSNA dataset is strongly imbalanced."
    )
    return model, framework, error
