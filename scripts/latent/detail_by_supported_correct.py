"""
PCA/UMAP visualization with 4-class labels:
  (supported=T/F) x (answer_correct=T/F)

Output files: {method}_layer_{L}_detail.png  (one per layer per method)

Usage:
    python detail_by_supported_correct.py \
        --latent_root /path/to/latent/results \
        --veri_file   /path/to/verification.json \
        --out_dir     /path/to/analysis_detail \
        --method pca|umap|both
"""

import argparse
import json
import os

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap as umap_lib


# ── labels ──────────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "sup✓ ans✓": "#2ca02c",   # green
    "sup✓ ans✗": "#98df8a",   # light green
    "sup✗ ans✓": "#d62728",   # red
    "sup✗ ans✗": "#ff9896",   # light red
}


SKIP_BOUNDARIES_PREFIXES = ("think_dense_token_", "think_token_")
SKIP_BOUNDARIES_EXACT = {"pre_generation", "think_first_content"}


BOUNDARY_DISPLAY = {
    "think_marker":    "<think>",
    "think_end":       "</think>",
    "search_before":   "<search>",
    "search_query_end":"</search>",
    "answer_before":   "<answer>",
    "answer_end":      "</answer>",
}


def _should_skip(boundary: str) -> bool:
    return boundary in SKIP_BOUNDARIES_EXACT or any(boundary.startswith(p) for p in SKIP_BOUNDARIES_PREFIXES)


def _display(boundary: str) -> str:
    return BOUNDARY_DISPLAY.get(boundary, boundary)


def normalize(s) -> str:
    return str(s).strip().lower() if s is not None else ""


def answer_correct(item: dict) -> bool:
    gold = normalize(item.get("answer"))
    aliases = [normalize(a) for a in item.get("answer_aliases", [])]
    pred = normalize(item.get("predicted_answer"))
    return pred == gold or pred in aliases


def make_label(supported: bool | None, correct: bool | None) -> str:
    s = "sup✓" if supported else "sup✗"
    a = "ans✓" if correct else "ans✗"
    return f"{s} {a}"


# ── data loading ─────────────────────────────────────────────────────────────
def load_label_map(veri_path: str) -> dict:
    """Returns {index: label_str}."""
    with open(veri_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for item in data:
        idx = item.get("index")
        verifs = item.get("triplet_verification", [])
        supported = all(v.get("supported", False) for v in verifs) if verifs else None
        correct = answer_correct(item)
        result[idx] = make_label(supported, correct)
    return result


def load_summary(root: str) -> list:
    path = os.path.join(root, "summary.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_vectors_and_labels(summary_rows: list, label_map: dict):
    by_layer: dict = {}
    labels: dict = {}
    boundaries: dict = {}

    for row in summary_rows:
        idx = row.get("index")
        label = label_map.get(idx, "unknown")

        for turn_info in row.get("latent_trace", []):
            for snapshot in turn_info.get("snapshots", []):
                tensor_path = snapshot.get("tensor_path")
                if not tensor_path or not os.path.exists(tensor_path):
                    continue
                boundary = snapshot.get("boundary", "")
                if _should_skip(boundary):
                    continue
                try:
                    payload = torch.load(tensor_path, map_location="cpu")
                except Exception:
                    continue
                for layer_key, tensor in payload.get("layers", {}).items():
                    layer_str = str(layer_key)
                    vec = tensor.float().numpy()
                    by_layer.setdefault(layer_str, []).append(vec)
                    labels.setdefault(layer_str, []).append(label)
                    boundaries.setdefault(layer_str, []).append(boundary)

    return by_layer, labels, boundaries


# ── projection ───────────────────────────────────────────────────────────────
def reduce_dims(matrix_scaled: np.ndarray, method: str) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=42).fit_transform(matrix_scaled)
    elif method == "umap":
        return umap_lib.UMAP(n_components=2, random_state=42, n_jobs=1).fit_transform(matrix_scaled)
    else:
        raise ValueError(f"Unknown method: {method}")


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_detail(
    vectors: list,
    label_list: list,
    layer: str,
    out_dir: str,
    method: str,
):
    matrix = np.stack(vectors).astype(np.float32)
    label_arr = np.array(label_list)

    coords = reduce_dims(StandardScaler().fit_transform(matrix), method)

    present_labels = [l for l in LABEL_COLORS if (label_arr == l).any()]
    counts = {l: int((label_arr == l).sum()) for l in present_labels}

    fig, ax = plt.subplots(figsize=(9, 7))
    for lbl in present_labels:
        mask = label_arr == lbl
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=14, alpha=0.55,
            color=LABEL_COLORS[lbl],
            label=f"{lbl} (n={counts[lbl]})",
        )

    method_label = method.upper()
    ax.set_title(
        f"{method_label} of boundary hidden states — layer {layer}\n"
        f"4-class: supported × answer_correct",
        fontsize=11,
    )
    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")
    ax.legend(markerscale=1.8, fontsize=9, frameon=False)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{method}_layer_{layer}_detail.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_boundary_by_group(
    vectors: list,
    label_list: list,
    boundary_list: list,
    layer: str,
    out_dir: str,
    method: str,
):
    """2x2 subplots: one per 4-class group, colored by boundary tag."""
    matrix = np.stack(vectors).astype(np.float32)
    label_arr = np.array(label_list)
    boundary_arr = np.array(boundary_list)

    coords = reduce_dims(StandardScaler().fit_transform(matrix), method)

    all_boundaries = sorted(set(boundary_arr))
    cmap = plt.get_cmap("tab10")
    b_colors = {b: cmap(i % 10) for i, b in enumerate(all_boundaries)}

    group_order = list(LABEL_COLORS.keys())
    method_label = method.upper()

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=False, sharey=False)
    axes_flat = axes.flatten()

    for ax, lbl in zip(axes_flat, group_order):
        mask = label_arr == lbl
        if not mask.any():
            ax.set_title(f"{lbl}\n(no data)")
            continue
        for b in all_boundaries:
            bm = mask & (boundary_arr == b)
            if bm.any():
                ax.scatter(coords[bm, 0], coords[bm, 1],
                           s=12, alpha=0.55, color=b_colors[b], label=_display(b))
        ax.set_title(f"{lbl}  (n={int(mask.sum())})")
        ax.set_xlabel(f"{method_label} 1")
        ax.set_ylabel(f"{method_label} 2")
        ax.legend(markerscale=1.4, fontsize=7, frameon=False)

    plt.suptitle(f"{method_label} by boundary tag — layer {layer}  (4-class)", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{method}_layer_{layer}_boundary_by_group.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_root", required=True)
    parser.add_argument("--veri_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--layers", nargs="*", default=None)
    parser.add_argument("--method", choices=["pca", "umap", "both"], default="both")
    args = parser.parse_args()

    methods = ["pca", "umap"] if args.method == "both" else [args.method]

    latent_root = os.path.abspath(args.latent_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading labels...")
    label_map = load_label_map(args.veri_file)
    from collections import Counter
    print("  label distribution:", dict(Counter(label_map.values())))

    print("Loading summary...")
    summary_rows = load_summary(latent_root)
    print(f"  {len(summary_rows)} samples")

    print("Collecting latent vectors...")
    by_layer, labels, boundaries = collect_vectors_and_labels(summary_rows, label_map)

    layers = args.layers or sorted(by_layer.keys(), key=lambda x: int(x))
    print(f"Layers: {layers}, methods: {methods}")

    plots = []
    for layer in layers:
        vecs = by_layer.get(layer)
        if not vecs:
            continue
        print(f"  Layer {layer}: {len(vecs)} vectors")
        for method in methods:
            p = plot_detail(vecs, labels[layer], layer, out_dir, method)
            p2 = plot_boundary_by_group(vecs, labels[layer], boundaries[layer], layer, out_dir, method)
            plots.extend([p, p2])

    print(f"\nDone. {len(plots)} plots saved to {out_dir}")


if __name__ == "__main__":
    main()
