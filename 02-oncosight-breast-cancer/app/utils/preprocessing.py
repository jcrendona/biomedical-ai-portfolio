from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


SEER_COLUMNS = [
    "Age",
    "Race",
    "Marital Status",
    "T Stage",
    "N Stage",
    "6th Stage",
    "Grade",
    "Tumor Size",
    "Estrogen Status",
    "Progesterone Status",
    "Regional Node Examined",
    "Reginol Node Positive",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def app_root() -> Path:
    return Path(__file__).resolve().parents[1]


def candidate_model_dirs() -> list[Path]:
    root = project_root()
    return [
        app_root() / "models",
        root / "models",
        root / "models" / "rsna",
        root / "models" / "pcam",
        root / "models" / "seer",
    ]


def find_model_file(keywords: list[str], suffixes: tuple[str, ...]) -> Path | None:
    for folder in candidate_model_dirs():
        if not folder.exists():
            continue
        candidates = []
        for suffix in suffixes:
            candidates.extend(folder.rglob(f"*{suffix}"))
        for path in sorted(candidates):
            name = path.name.lower()
            if any(keyword.lower() in name for keyword in keywords):
                return path
    return None


def find_saved_model_dir(keywords: list[str]) -> Path | None:
    for folder in candidate_model_dirs():
        if not folder.exists():
            continue
        for pb in folder.rglob("saved_model.pb"):
            parent = pb.parent
            if any(keyword.lower() in str(parent).lower() for keyword in keywords):
                return parent
    return None


def load_uploaded_image(uploaded_file: Any) -> Image.Image:
    name = getattr(uploaded_file, "name", "").lower()
    data = uploaded_file.read()
    uploaded_file.seek(0)
    if name.endswith(".dcm"):
        try:
            import pydicom

            ds = pydicom.dcmread(BytesIO(data))
            arr = ds.pixel_array.astype(np.float32)
            arr -= arr.min()
            if arr.max() > 0:
                arr /= arr.max()
            arr = (arr * 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Could not read DICOM image: {exc}") from exc
    return Image.open(BytesIO(data)).convert("RGB")


def preprocess_image(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    resized = image.resize(size)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def image_to_uint8(image: Image.Image, size: tuple[int, int] | None = None) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if size:
        image = image.resize(size)
    return np.asarray(image).astype(np.uint8)


def seer_form_to_frame(values: dict[str, Any]) -> pd.DataFrame:
    row = {
        "Age": values["age"],
        "Race": values["race"],
        "Marital Status": values["marital_status"],
        "T Stage": values["t_stage"],
        "N Stage": values["n_stage"],
        "6th Stage": values["sixth_stage"],
        "Grade": values["grade"],
        "Tumor Size": values["tumor_size"],
        "Estrogen Status": values["estrogen_status"],
        "Progesterone Status": values["progesterone_status"],
        "Regional Node Examined": values["regional_node_examined"],
        "Reginol Node Positive": values["regional_node_positive"],
    }
    return pd.DataFrame([row])


def fallback_seer_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    mappings = {
        "Race": {"White": 0, "Black": 1, "Other": 2},
        "Marital Status": {"Married": 0, "Single": 1, "Divorced": 2, "Widowed": 3, "Separated": 4},
        "T Stage": {"T1": 1, "T2": 2, "T3": 3, "T4": 4},
        "N Stage": {"N1": 1, "N2": 2, "N3": 3},
        "6th Stage": {"IIA": 2, "IIB": 3, "IIIA": 4, "IIIB": 5, "IIIC": 6},
        "Grade": {
            "Well differentiated": 1,
            "Moderately differentiated": 2,
            "Poorly differentiated": 3,
            "Undifferentiated": 4,
        },
        "Estrogen Status": {"Positive": 1, "Negative": 0},
        "Progesterone Status": {"Positive": 1, "Negative": 0},
    }
    out = df.copy()
    for col, mapping in mappings.items():
        if col in out:
            out[col] = out[col].map(mapping).fillna(0)
    return out.astype(float)


def typical_seer_reference() -> pd.DataFrame:
    rows = []
    for age in [35, 45, 55, 65, 75, 85]:
        rows.append(
            {
                "Age": age,
                "Race": 0,
                "Marital Status": 0,
                "T Stage": 1 + (age > 60),
                "N Stage": 1,
                "6th Stage": 2 + (age > 70),
                "Grade": 2,
                "Tumor Size": 10 + (age - 35) * 0.7,
                "Estrogen Status": 1,
                "Progesterone Status": 1,
                "Regional Node Examined": 12,
                "Reginol Node Positive": 1 + (age > 70),
            }
        )
    return pd.DataFrame(rows)
