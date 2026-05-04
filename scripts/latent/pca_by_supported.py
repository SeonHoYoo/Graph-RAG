"""
PCA/UMAP visualization of latent hidden states, colored by verification_result supported label.

For each layer, produces scatter plots where:
  - supported=True  (all triplets supported) → one color
  - supported=False (any triplet unsupported) → another color

Usage:
    python pca_by_supported.py \
        --latent_root /path/to/latent/results \
        --veri_file   /path/to/verification.json \
        --out_dir     /path/to/output \
        [--method pca|umap|both]
"""

import argparse
import json
import os
import re

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap as umap_lib


SAMPLE_RE = re.compile(r"^sample_(?P<index>\d+)__uid_(?P<uid>[^_]+)__")

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


def _normalize(s) -> str:
    return str(s).strip().lower() if s is not None else ""


def load_supported_map(veri_path: str) -> dict:
    """Returns {index: (supported, answer_correct)} tuples."""
    with open(veri_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for item in data:
        idx = item.get("index")
        verifs = item.get("triplet_verification", [])
        supported = all(v.get("supported", False) for v in verifs) if verifs else None
        gold = _normalize(item.get("answer"))
        aliases = [_normalize(a) for a in item.get("answer_aliases", [])]
        pred = _normalize(item.get("predicted_answer"))
        correct = pred == gold or pred in aliases
        result[idx] = (supported, correct)
    return result


def load_summary(root: str) -> list:
    path = os.path.join(root, "summary.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_vectors_and_labels(summary_rows: list, supported_map: dict):
    """
    Returns:
        by_layer: dict[str, list[np.ndarray]]  — vectors per layer
        labels:   dict[str, list[bool|None]]   — supported label per vector
        boundaries: dict[str, list[str]]        — boundary tag per vector
    """
    by_layer: dict = {}
    labels: dict = {}
    boundaries: dict = {}

    for row in summary_rows:
        idx = row.get("index")
        supported, correct = supported_map.get(idx, (None, None))

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
                    labels.setdefault(layer_str, []).append((supported, correct))
                    boundaries.setdefault(layer_str, []).append(boundary)

    return by_layer, labels, boundaries


def reduce_dims(matrix_scaled: np.ndarray, method: str) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=42).fit_transform(matrix_scaled)
    elif method == "umap":
        reducer = umap_lib.UMAP(n_components=2, random_state=42, n_jobs=1)
        return reducer.fit_transform(matrix_scaled)
    else:
        raise ValueError(f"Unknown method: {method}")


GROUP_COLORS = {
    (True,  True):  "#2ca02c",   # sup✓ ans✓  green
    (True,  False): "#98df8a",   # sup✓ ans✗  light green
    (False, True):  "#d62728",   # sup✗ ans✓  red
    (False, False): "#ff9896",   # sup✗ ans✗  light red
}
GROUP_LABELS = {
    (True,  True):  "sup✓ ans✓",
    (True,  False): "sup✓ ans✗",
    (False, True):  "sup✗ ans✓",
    (False, False): "sup✗ ans✗",
}


def plot_by_supported(
    vectors: list,
    label_list: list,
    boundary_list: list,
    layer: str,
    out_dir: str,
    method: str,
):
    matrix = np.stack(vectors).astype(np.float32)
    coords = reduce_dims(StandardScaler().fit_transform(matrix), method)

    method_label = method.upper()
    fig, ax = plt.subplots(figsize=(9, 7))
    for key, lbl in GROUP_LABELS.items():
        mask = np.array([l == key for l in label_list])
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=14, alpha=0.55, color=GROUP_COLORS[key],
                       label=f"{lbl} (n={mask.sum()})")
    none_mask = np.array([l[0] is None for l in label_list])
    if none_mask.any():
        ax.scatter(coords[none_mask, 0], coords[none_mask, 1],
                   s=14, alpha=0.40, color="#AAAAAA", label="no triplet data")

    ax.set_title(f"{method_label} of boundary hidden states — layer {layer}\n"
                 f"4-class: supported × answer_correct", fontsize=11)
    ax.set_xlabel(f"{method_label} 1")
    ax.set_ylabel(f"{method_label} 2")
    ax.legend(markerscale=1.8, fontsize=9, frameon=False)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{method}_layer_{layer}_supported.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_by_boundary_within_group(
    vectors: list,
    label_list: list,
    boundary_list: list,
    layer: str,
    out_dir: str,
    method: str,
):
    matrix = np.stack(vectors).astype(np.float32)
    boundary_arr = np.array(boundary_list)
    coords = reduce_dims(StandardScaler().fit_transform(matrix), method)

    all_boundaries = sorted(set(boundary_arr))
    cmap = plt.get_cmap("tab10")
    b_colors = {b: cmap(i % 10) for i, b in enumerate(all_boundaries)}

    group_order = [(True, True), (True, False), (False, True), (False, False)]
    method_label = method.upper()

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=False, sharey=False)
    for ax, key in zip(axes.flatten(), group_order):
        lbl = GROUP_LABELS[key]
        mask = np.array([l == key for l in label_list])
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
        ax.legend(markerscale=1.5, fontsize=7, frameon=False)

    plt.suptitle(f"{method_label} by boundary tag — layer {layer}  (4-class)", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{method}_layer_{layer}_boundary_by_group.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


# kept for backward compatibility
def plot_pca_by_supported(vectors, label_list, boundary_list, layer, out_dir):
    return plot_by_supported(vectors, label_list, boundary_list, layer, out_dir, "pca")

def plot_pca_by_boundary_within_group(vectors, label_list, boundary_list, layer, out_dir):
    return plot_by_boundary_within_group(vectors, label_list, boundary_list, layer, out_dir, "pca")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_root", required=True)
    parser.add_argument("--veri_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--layers", nargs="*", default=None,
                        help="Layers to plot (default: all found in data)")
    parser.add_argument("--method", choices=["pca", "umap", "both"], default="pca",
                        help="Dimensionality reduction method")
    args = parser.parse_args()

    methods = ["pca", "umap"] if args.method == "both" else [args.method]

    latent_root = os.path.abspath(args.latent_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading verification labels...")
    supported_map = load_supported_map(args.veri_file)
    from collections import Counter
    print("  label distribution:", dict(Counter(supported_map.values())))

    print("Loading summary...")
    summary_rows = load_summary(latent_root)
    print(f"  {len(summary_rows)} samples in summary")

    print("Collecting latent vectors (this may take a while)...")
    by_layer, labels, boundaries = collect_vectors_and_labels(summary_rows, supported_map)

    layers = args.layers or sorted(by_layer.keys(), key=lambda x: int(x))
    print(f"Layers to plot: {layers}, methods: {methods}")

    plots = []
    for layer in layers:
        vecs = by_layer.get(layer)
        if not vecs:
            print(f"  Layer {layer}: no vectors, skipping.")
            continue
        print(f"  Layer {layer}: {len(vecs)} vectors")

        # save PCA coordinates CSV (always, regardless of method)
        matrix = np.stack(vecs).astype(np.float32)
        matrix_scaled = StandardScaler().fit_transform(matrix)
        pca_coords = PCA(n_components=2, random_state=42).fit_transform(matrix_scaled)
        import csv
        csv_path = os.path.join(out_dir, f"pca_coords_layer_{layer}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["pca1", "pca2", "boundary", "supported", "correct"])
            for i, (pca1, pca2) in enumerate(pca_coords):
                sup, cor = labels[layer][i]
                writer.writerow([pca1, pca2, boundaries[layer][i],
                                 "" if sup is None else int(sup),
                                 "" if cor is None else int(cor)])
        print(f"  Saved coords: {csv_path}")

        for method in methods:
            p1 = plot_by_supported(vecs, labels[layer], boundaries[layer], layer, out_dir, method)
            p2 = plot_by_boundary_within_group(vecs, labels[layer], boundaries[layer], layer, out_dir, method)
            plots.extend([p1, p2])

    print(f"\nDone. {len(plots)} plots saved to {out_dir}")


if __name__ == "__main__":
    main()
