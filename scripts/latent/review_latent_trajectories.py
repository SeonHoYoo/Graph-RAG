import argparse
import html
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_summary(root: str) -> List[Dict[str, Any]]:
    summary_path = os.path.join(root, "summary.json")
    with open(summary_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_vectors_for_sample(row: Dict[str, Any], layer: str) -> Tuple[pd.DataFrame, np.ndarray]:
    records = []
    vectors = []
    for turn_info in row.get("latent_trace", []):
        turn = turn_info.get("turn")
        for snapshot in turn_info.get("snapshots", []):
            tensor_path = snapshot.get("tensor_path")
            if not tensor_path or not os.path.exists(tensor_path):
                continue
            payload = torch.load(tensor_path, map_location="cpu")
            layers = payload.get("layers", {})
            if layer not in layers:
                continue
            vector = layers[layer].float().numpy()
            records.append(
                {
                    "turn": turn,
                    "snapshot_index": snapshot.get("snapshot_index"),
                    "boundary": snapshot.get("boundary"),
                    "boundary_start_generated_token": snapshot.get("boundary_start_generated_token"),
                    "tensor_path": tensor_path,
                }
            )
            vectors.append(vector)

    if not records:
        return pd.DataFrame(), np.empty((0, 0), dtype=np.float32)
    df = pd.DataFrame(records).sort_values(["turn", "snapshot_index"]).reset_index(drop=True)
    return df, np.stack(vectors).astype(np.float32)


def fit_global_pca(rows: List[Dict[str, Any]], layer: str, max_points: int) -> Tuple[StandardScaler, PCA]:
    all_vectors = []
    for row in rows:
        _, vectors = load_vectors_for_sample(row, layer)
        if vectors.size:
            all_vectors.append(vectors)
    if not all_vectors:
        raise ValueError(f"No vectors found for layer {layer}")

    matrix = np.concatenate(all_vectors, axis=0)
    if len(matrix) > max_points:
        rng = np.random.default_rng(42)
        matrix = matrix[rng.choice(len(matrix), size=max_points, replace=False)]
    scaler = StandardScaler().fit(matrix)
    pca = PCA(n_components=2, random_state=42).fit(scaler.transform(matrix))
    return scaler, pca


def trajectory_metrics(coords: np.ndarray, vectors: np.ndarray) -> Dict[str, float]:
    if len(vectors) < 2:
        return {
            "num_points": float(len(vectors)),
            "path_length_hidden": np.nan,
            "direct_distance_hidden": np.nan,
            "tortuosity_hidden": np.nan,
            "mean_step_cosine": np.nan,
            "path_length_pca": np.nan,
            "direct_distance_pca": np.nan,
            "tortuosity_pca": np.nan,
        }

    hidden_steps = np.linalg.norm(np.diff(vectors, axis=0), axis=1)
    hidden_direct = float(np.linalg.norm(vectors[-1] - vectors[0]))
    hidden_path = float(hidden_steps.sum())
    step_cosines = []
    for prev, cur in zip(vectors[:-1], vectors[1:]):
        denom = np.linalg.norm(prev) * np.linalg.norm(cur)
        if denom:
            step_cosines.append(float(np.dot(prev, cur) / denom))

    pca_steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    pca_direct = float(np.linalg.norm(coords[-1] - coords[0]))
    pca_path = float(pca_steps.sum())
    return {
        "num_points": float(len(vectors)),
        "path_length_hidden": hidden_path,
        "direct_distance_hidden": hidden_direct,
        "tortuosity_hidden": hidden_path / hidden_direct if hidden_direct else np.nan,
        "mean_step_cosine": float(np.mean(step_cosines)) if step_cosines else np.nan,
        "path_length_pca": pca_path,
        "direct_distance_pca": pca_direct,
        "tortuosity_pca": pca_path / pca_direct if pca_direct else np.nan,
    }


BOUNDARY_COLORS = {
    # new-style
    "pre_generation":      "#999999",
    "think_marker":        "#1565C0",
    "think_first_content": "#1E88E5",
    "think_end":           "#FF6D00",
    "search_before":       "#2E7D32",
    "search_query_end":    "#00897B",
    "answer_before":       "#C62828",
    "answer_end":          "#AD1457",
    # legacy angle-bracket style
    "<think>":             "#1565C0",
    "</think>":            "#FF6D00",
    "<search>":            "#2E7D32",
    "</search>":           "#00897B",
    "<information>":       "#6A1B9A",
    "</information>":      "#4527A0",
    "<answer>":            "#C62828",
    "</answer>":           "#AD1457",
}

BOUNDARY_DISPLAY = {
    "pre_generation":      "pre",
    "think_marker":        "<think>",
    "think_first_content": "<think>+1",
    "think_end":           "</think>",
    "search_before":       "<search>",
    "search_query_end":    "</search>",
    "answer_before":       "<answer>",
    "answer_end":          "</answer>",
}

SKIP_PREFIXES = ("think_dense_token_", "think_token_")
SKIP_EXACT = {"pre_generation", "think_first_content"}


def plot_sample_trajectory(
    sample_df: pd.DataFrame,
    coords: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    # filter out dense/offset think tokens and exact skips
    keep = ~sample_df["boundary"].str.startswith(SKIP_PREFIXES, na=False) & ~sample_df["boundary"].isin(SKIP_EXACT)
    plot_df = sample_df[keep].copy()
    plot_coords = coords[plot_df.index]

    plt.figure(figsize=(8, 6))
    if len(plot_coords) > 1:
        plt.plot(plot_coords[:, 0], plot_coords[:, 1], color="black", linewidth=0.9, alpha=0.5, zorder=1)
    boundary_counts: dict = {}
    for i, (_, row) in enumerate(plot_df.iterrows()):
        boundary = row["boundary"]
        color = BOUNDARY_COLORS.get(boundary, "#333333")
        plt.scatter(plot_coords[i, 0], plot_coords[i, 1], s=100, color=color, zorder=3, edgecolors="white", linewidths=0.5)
        display = BOUNDARY_DISPLAY.get(boundary, boundary)
        within = boundary_counts.get(boundary, 0)
        boundary_counts[boundary] = within + 1
        label = f'{i}:{within}:{display}'
        plt.text(plot_coords[i, 0], plot_coords[i, 1], label, fontsize=7, ha="left", va="bottom", color="black")
    plt.title(title)
    plt.xlabel("global PCA 1")
    plt.ylabel("global PCA 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def shorten(value: Optional[str], limit: int) -> str:
    if value is None:
        return ""
    value = str(value)
    return value if len(value) <= limit else value[:limit] + "\n...[truncated]"


def write_review_html(rows: List[Dict[str, Any]], out_path: str, title: str) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;line-height:1.35;color:#222}",
        ".sample{border-top:1px solid #ccc;padding:20px 0;display:grid;grid-template-columns:420px 1fr;gap:22px}",
        "img{width:420px;max-width:100%;border:1px solid #ddd}",
        "pre{white-space:pre-wrap;background:#f7f7f7;padding:10px;border:1px solid #ddd;max-height:360px;overflow:auto}",
        "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:4px 6px}",
        ".label{font-weight:bold;color:#777}",
        "</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        "<p>Manual label columns are left blank in the CSV. Use the plot plus response text to label hallucination or wrong reasoning.</p>",
    ]
    for row in rows:
        parts.append("<div class='sample'>")
        parts.append(f"<div><img src='{html.escape(os.path.basename(row['plot_path']))}'></div>")
        parts.append("<div>")
        parts.append(f"<h2>sample {html.escape(str(row['index']))} | uid {html.escape(str(row['uid']))}</h2>")
        parts.append("<table>")
        for key in [
            "num_points",
            "boundary_sequence",
            "path_length_hidden",
            "direct_distance_hidden",
            "tortuosity_hidden",
            "mean_step_cosine",
            "predicted_answer",
        ]:
            parts.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(row.get(key, '')))}</td></tr>")
        parts.append("</table>")
        parts.append("<p class='label'>question</p>")
        parts.append(f"<pre>{html.escape(shorten(row.get('question'), 1200))}</pre>")
        parts.append("<p class='label'>full_response</p>")
        parts.append(f"<pre>{html.escape(shorten(row.get('full_response'), 6000))}</pre>")
        parts.append("</div></div>")
    parts.append("</body></html>")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create per-sample latent trajectory outputs for manual hallucination/reasoning review.")
    parser.add_argument("--latent_root", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--layer", default="28")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_pca_fit_points", type=int, default=5000)
    args = parser.parse_args()

    latent_root = os.path.abspath(args.latent_root)
    out_dir = os.path.abspath(args.out_dir or os.path.join(latent_root, f"trajectory_review_layer_{args.layer}"))
    os.makedirs(out_dir, exist_ok=True)

    rows = load_summary(latent_root)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    scaler, pca = fit_global_pca(rows, args.layer, args.max_pca_fit_points)
    review_rows = []
    point_rows = []

    for row in rows:
        sample_df, vectors = load_vectors_for_sample(row, args.layer)
        if sample_df.empty:
            continue
        coords = pca.transform(scaler.transform(vectors))
        metrics = trajectory_metrics(coords, vectors)
        boundary_sequence = " -> ".join(sample_df["boundary"].astype(str).tolist())
        index = row.get("index")
        uid = row.get("uid")
        plot_name = f"sample_{index}__uid_{uid}__layer_{args.layer}.png".replace("/", "_")
        plot_path = os.path.join(out_dir, plot_name)
        plot_sample_trajectory(sample_df, coords, plot_path, f"sample {index}, layer {args.layer}")

        for point_idx, point in sample_df.iterrows():
            point_rows.append(
                {
                    "index": index,
                    "uid": uid,
                    "layer": args.layer,
                    "point_order": point_idx,
                    "turn": point["turn"],
                    "snapshot_index": point["snapshot_index"],
                    "boundary": point["boundary"],
                    "x": coords[point_idx, 0],
                    "y": coords[point_idx, 1],
                    "tensor_path": point["tensor_path"],
                }
            )

        review_row = {
            "index": index,
            "uid": uid,
            "layer": args.layer,
            "manual_label": "",
            "manual_notes": "",
            "question": row.get("question"),
            "predicted_answer": row.get("predicted_answer"),
            "num_turns": row.get("num_turns"),
            "boundary_sequence": boundary_sequence,
            "plot_path": plot_path,
            "full_response": row.get("full_response"),
            **metrics,
        }
        review_rows.append(review_row)

    review_df = pd.DataFrame(review_rows)
    if not review_df.empty:
        review_df = review_df.sort_values(["tortuosity_hidden", "path_length_hidden"], ascending=False, na_position="last")
    review_csv = os.path.join(out_dir, "manual_trajectory_review.csv")
    points_csv = os.path.join(out_dir, "trajectory_points.csv")
    html_path = os.path.join(out_dir, "manual_trajectory_review.html")
    review_df.to_csv(review_csv, index=False)
    pd.DataFrame(point_rows).to_csv(points_csv, index=False)
    write_review_html(review_df.to_dict("records"), html_path, f"Latent trajectory manual review, layer {args.layer}")

    print(
        json.dumps(
            {
                "out_dir": out_dir,
                "layer": args.layer,
                "num_samples": len(review_df),
                "review_csv": review_csv,
                "points_csv": points_csv,
                "review_html": html_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
