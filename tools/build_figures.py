"""Build every portfolio figure from the notebooks and metric files in this repo.

All numbers plotted here are read from executed notebook outputs or from the
metrics JSON files exported during training. Nothing is re-estimated or invented.

Usage (from the repo root):
    swift tools/pdf2png.swift 02-oncosight-breast-cancer/docs/oncosight_demo_walkthrough.pdf build/onco 3
    python tools/build_figures.py

Outputs go to each project's `figures/` folder.
"""
from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CXR = ROOT / "01-chest-xray-multitask"
ONCO = ROOT / "02-oncosight-breast-cancer"
KID = ROOT / "03-kidney-ct-eda"
ONCO_PAGES = ROOT / "build" / "onco"  # rendered by tools/pdf2png.swift

INK = "#14212b"
MUTED = "#5b6670"
GRID = "#e3e8ec"
PRIMARY = "#0b5d7a"
ACCENT = "#c2410c"
SERIES = ["#0b5d7a", "#3f8fb0", "#9cc5d6", "#c2410c"]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": "#9aa5ad",
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "savefig.facecolor": "white",
})


# --------------------------------------------------------------------------- helpers
def notebook_images(nb_path: Path) -> dict[tuple[int, int], Image.Image]:
    nb = json.loads(nb_path.read_text())
    imgs = {}
    for i, cell in enumerate(nb["cells"]):
        for j, out in enumerate(cell.get("outputs", [])):
            data = out.get("data", {})
            if "image/png" in data:
                imgs[(i, j)] = Image.open(io.BytesIO(base64.b64decode(data["image/png"]))).convert("RGB")
    return imgs


def notebook_text(nb_path: Path, cell: int) -> str:
    nb = json.loads(nb_path.read_text())
    return "".join("".join(o.get("text", "")) for o in nb["cells"][cell].get("outputs", []))


def axes_crop(img: Image.Image, box: tuple[int, int, int, int], inset: int = 2) -> Image.Image:
    """Crop the pixel area inside a matplotlib axes (box found by inspecting the figure)."""
    x0, y0, x1, y1 = box
    return img.crop((x0 + inset, y0 + inset, x1 - inset, y1 - inset))


def viridis_mask_to_binary(img: Image.Image) -> np.ndarray:
    """Notebook masks were drawn with the default viridis map: yellow = lung."""
    a = np.asarray(img).astype(int)
    return (a[..., 0] > 150) & (a[..., 1] > 150) & (a[..., 2] < 120)


def panel_label(ax, text):
    ax.text(0.025, 0.975, text, transform=ax.transAxes, ha="left", va="top", color="white",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.3", facecolor="#14212b", edgecolor="none", alpha=0.85))


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


# --------------------------------------------------------------------------- chest X-ray
def chest_xray_figures():
    nb_tf = CXR / "notebooks" / "01_tfrecord_dataset_and_boundary_weights.ipynb"
    nb_mb = CXR / "notebooks" / "03_mobilenet_multitask_gradcam.ipynb"
    nb_cmp = CXR / "notebooks" / "04_chexpert_model_comparison.ipynb"

    mb = notebook_images(nb_mb)
    tf = notebook_images(nb_tf)

    cxr = axes_crop(mb[(9, 1)], (45, 129, 274, 357))
    mask = viridis_mask_to_binary(axes_crop(mb[(9, 1)], (316, 129, 544, 357)))
    w_mask = viridis_mask_to_binary(axes_crop(tf[(7, 1)], (33, 10, 252, 229)))
    weight = np.asarray(axes_crop(tf[(5, 1)], (7, 6, 225, 224)).convert("L")).astype(float)
    # Index 600 lies in the augmented half of x_train (rotated -10 degrees with its mask).
    aug = np.asarray(axes_crop(mb[(15, 1)], (45, 129, 274, 357)).convert("L"))
    aug_mask = viridis_mask_to_binary(axes_crop(mb[(15, 1)], (316, 129, 544, 357)))
    # Note: the notebook's Grad-CAM cell uses the ImageNet MobileNet head, not the
    # trained classifier, so it is deliberately not used as model evidence here.

    # Figure 1: pipeline evidence --------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    gray = np.asarray(cxr.convert("L"))
    axes[0].imshow(gray, cmap="gray")
    panel_label(axes[0], "A")
    axes[0].set_title("Input radiograph", loc="left")

    axes[1].imshow(gray, cmap="gray")
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask] = (0.04, 0.55, 0.75, 0.35)
    axes[1].imshow(overlay)
    axes[1].contour(mask, levels=[0.5], colors=["#38bdf8"], linewidths=1.4)
    panel_label(axes[1], "B")
    axes[1].set_title("Lung mask (ground truth)", loc="left")

    im = axes[2].imshow(weight, cmap=LinearSegmentedColormap.from_list("w", ["#0b1720", "#3f8fb0", "#f8fafc"]))
    axes[2].contour(w_mask, levels=[0.5], colors=[ACCENT], linewidths=1.0)
    panel_label(axes[2], "C")
    axes[2].set_title("Boundary-aware pixel weights", loc="left")

    axes[3].imshow(aug, cmap="gray")
    axes[3].contour(aug_mask, levels=[0.5], colors=["#38bdf8"], linewidths=1.4)
    panel_label(axes[3], "D")
    axes[3].set_title("Augmentation: image + mask rotated -10°", loc="left")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(w_pad=0.6)
    save(fig, CXR / "figures" / "cxr_pipeline_panels.png")

    # Thumbnail used by the website card
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(gray, cmap="gray"); ax.imshow(overlay)
    ax.contour(mask, levels=[0.5], colors=["#38bdf8"], linewidths=1.6)
    ax.axis("off")
    save(fig, CXR / "figures" / "cxr_thumbnail.png")

    # Figure 2: AUC heatmap from the executed comparison notebook ------------------
    cells = {
        6: "DenseNet-121 · U-Ones", 7: "DenseNet-121 · U-Multi",
        8: "CRAL att-1 · U-Ones", 9: "CRAL att-1 · U-Multi",
        10: "CRAL att-2 · U-Ones", 11: "CRAL att-2 · U-Multi",
        12: "AG-CNN global · U-Ones", 14: "AG-CNN local · U-Ones",
        13: "AG-CNN fusion · U-Ones", 5: "Weighted ensemble (0.3/0.3/0.4)",
    }
    rows, means, classes = [], [], None
    for cell, name in cells.items():
        txt = notebook_text(nb_cmp, cell)
        means.append(float(re.search(r"AUC: ([0-9.]+)", txt).group(1)))
        pairs = re.findall(r"\d+\s+([A-Za-z_]+)\s+([0-9.]+)", txt)
        classes = [p[0].replace("_", " ") for p in pairs]
        rows.append([float(p[1]) for p in pairs])
    auc = np.array(rows)

    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    cmap = LinearSegmentedColormap.from_list("auc", ["#fbfcfd", "#e4eff4", "#a9cfdf", "#4f97b8", "#0b5d7a", "#05303f"])
    hm = ax.imshow(auc, cmap=cmap, vmin=0.4, vmax=1.0, aspect="auto")
    for i in range(auc.shape[0]):
        for j in range(auc.shape[1]):
            v = auc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.2,
                    color="white" if v > 0.84 else INK)
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(cells)), [f"{n}   ({m:.3f})" for n, m in zip(cells.values(), means)])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Per-finding ROC-AUC on the CheXpert validation set (234 studies)   ·   row label = mean AUC", loc="left")
    cb = fig.colorbar(hm, ax=ax, fraction=0.025, pad=0.01)
    cb.outline.set_visible(False)
    cb.set_label("ROC-AUC")
    save(fig, CXR / "figures" / "chexpert_auc_heatmap.png")

    # Figure 3: mean AUC ranking ---------------------------------------------------
    order = np.argsort(means)
    names = np.array(list(cells.values()))[order]
    vals = np.array(means)[order]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    colors = [ACCENT if "ensemble" in n else PRIMARY for n in names]
    ax.barh(names, vals, color=colors, height=0.62)
    for y, v in enumerate(vals):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0.76, 0.83)
    ax.set_xlabel("Mean ROC-AUC across 13 findings (axis starts at 0.76 to show differences)")
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    ax.set_title("Model comparison, CheXpert validation set", loc="left")
    save(fig, CXR / "figures" / "chexpert_mean_auc.png")


# --------------------------------------------------------------------------- OncoSight
def oncosight_figures():
    seer = json.loads((ONCO / "app/models/seer/metrics.json").read_text())["metrics"]
    rsna = json.loads((ONCO / "app/models/rsna/metrics.json").read_text())

    # SEER clinical model comparison (held-out test split)
    models = [("dummy_prior", "Prior (baseline)"), ("logistic_regression_balanced", "Logistic regression"),
              ("hist_gradient_boosting", "Hist. gradient boosting"), ("random_forest_balanced", "Random forest (selected)")]
    metrics = [("roc_auc", "ROC-AUC"), ("recall", "Recall"), ("precision", "Precision"), ("f1", "F1")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(metrics))
    w = 0.2
    for k, (key, label) in enumerate(models):
        vals = [seer[key]["test"][m] for m, _ in metrics]
        color = ["#c7cdd2", "#9cc5d6", "#3f8fb0", PRIMARY][k]
        bars = ax.bar(x + (k - 1.5) * w, vals, w, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, [m for _, m in metrics])
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    n_test = sum(map(sum, seer["random_forest_balanced"]["test"]["confusion_matrix"]))
    ax.set_title(f"Clinical-context module: held-out test split (n = {n_test}), threshold tuned on validation", loc="left")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
    save(fig, ONCO / "figures" / "seer_model_comparison.png")

    # RSNA ResNet-50 validation history
    hist = rsna["resnet50_history"]
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(ep, [h["auc"] for h in hist], color=PRIMARY, lw=2, label="Validation ROC-AUC")
    ax.plot(ep, [h["recall"] for h in hist], color=ACCENT, lw=1.4, alpha=0.85, label="Validation recall (cancer class)")
    ax.axhline(0.5, color="#9aa5ad", lw=1, ls="--")
    ax.text(1, 0.49, "chance level", ha="left", va="top", fontsize=8, color=MUTED)
    ax.set_xlabel("Epoch"); ax.set_ylim(0.3, 0.9)
    ax.grid(color=GRID); ax.set_axisbelow(True)
    cm = rsna["resnet50_val"]["confusion_matrix"]
    ax.set_title(f"Mammography module (ResNet-50): validation set of {sum(map(sum, cm))} images, "
                 f"{cm[1][0] + cm[1][1]} cancer-positive", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    save(fig, ONCO / "figures" / "rsna_resnet50_history.png")

    # Screenshots from the recorded demo walkthrough (rendered PDF pages)
    if not ONCO_PAGES.exists():
        print("skip screenshots: run tools/pdf2png.swift first")
        return
    f = Image.open(ONCO_PAGES / "page1.png").size[0] / 1240

    def crop(page, box):
        return Image.open(ONCO_PAGES / page).convert("RGB").crop(tuple(int(v * f) for v in box))

    mammo = crop("page3.png", (487, 43, 1085, 641))
    gcam = crop("page5.png", (487, 43, 1085, 641))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, im, t, lab in zip(axes, [mammo, gcam], ["Uploaded mammogram (demo input)", "Grad-CAM, ResNet-50 output"], "AB"):
        ax.imshow(im); ax.axis("off"); ax.set_title(t, loc="left"); panel_label(ax, lab)
    fig.tight_layout()
    save(fig, ONCO / "figures" / "oncosight_gradcam.png")
    crop("page1.png", (43, 43, 1100, 840)).save(ONCO / "figures" / "oncosight_ui.png")
    crop("page6.png", (480, 85, 1100, 860)).save(ONCO / "figures" / "oncosight_fusion_summary.png")
    print("wrote OncoSight screenshots")


# --------------------------------------------------------------------------- kidney CT
def kidney_figures():
    nb = KID / "notebooks" / "kidney_ct_eda_patient_split.ipynb"
    imgs = notebook_images(nb)
    grid = imgs[(16, 0)]
    xs = [(10, 283), (307, 580), (604, 877), (901, 1174), (1198, 1471)]
    ys = [(36, 309), (323, 596), (611, 884), (898, 1171)]
    classes = ["Normal", "Cyst", "Tumor", "Stone"]
    fig, axes = plt.subplots(4, 5, figsize=(10, 8.4))
    for r, (y0, y1) in enumerate(ys):
        for c, (x0, x1) in enumerate(xs):
            ax = axes[r, c]
            ax.imshow(grid.crop((x0 + 1, y0 + 1, x1 - 1, y1 - 1)))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 0:
                ax.set_ylabel(classes[r], fontsize=11, fontweight="bold", color=INK, rotation=0, ha="right", va="center", labelpad=12)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    save(fig, KID / "figures" / "kidney_ct_samples.png")

    # Class distribution: parsed from the executed inventory cell
    txt = notebook_text(nb, 6)
    counts = {c: int(re.search(rf"{c}\s+(\d+)", txt).group(1)) for c in classes}
    wtxt = notebook_text(nb, 29)
    weights = {c: float(re.search(rf"{c}: ([0-9.]+)", wtxt).group(1)) for c in classes}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    total = sum(counts.values())
    bars = a1.bar(classes, [counts[c] for c in classes], color=SERIES)
    for b, c in zip(bars, classes):
        a1.text(b.get_x() + b.get_width() / 2, b.get_height() + 60,
                f"{counts[c]:,}  ({counts[c] / total:.0%})", ha="center", fontsize=9, color=INK)
    a1.set_ylim(0, max(counts.values()) * 1.15)
    a1.set_title(f"Images per class (n = {total:,})", loc="left")
    a1.grid(axis="y", color=GRID); a1.set_axisbelow(True)
    a2.barh(classes[::-1], [weights[c] for c in classes[::-1]], color=SERIES[::-1])
    for y, c in enumerate(classes[::-1]):
        a2.text(weights[c] + 0.03, y, f"{weights[c]:.2f}", va="center", fontsize=9, color=INK)
    a2.axvline(1, color="#9aa5ad", lw=1, ls="--")
    a2.set_xlim(0, 2.6)
    a2.set_title("Balanced class weights (training split)", loc="left")
    save(fig, KID / "figures" / "kidney_class_balance.png")

    imgs[(24, 0)].save(KID / "figures" / "kidney_pca.png")
    print("wrote", (KID / "figures" / "kidney_pca.png").relative_to(ROOT))


if __name__ == "__main__":
    chest_xray_figures()
    oncosight_figures()
    kidney_figures()
