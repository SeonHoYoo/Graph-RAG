import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


SAMPLE_RE = re.compile(r"^sample_(?P<index>.*?)__uid_(?P<uid>.*?)__q_(?P<qhash>.*)$")
TURN_RE = re.compile(r"^turn_(?P<turn>\d+)$")


def load_summary(root: str) -> Dict[str, Dict[str, Any]]:
    path = os.path.join(root, "summary.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    by_dir = {}
    for row in rows:
        trace_dir = row.get("latent_trace_dir")
        if trace_dir:
            by_dir[os.path.abspath(trace_dir)] = row
    return by_dir


def iter_metadata(root: str) -> Iterable[Tuple[str, str, str]]:
    for sample_name in sorted(os.listdir(root)):
        sample_dir = os.path.join(root, sample_name)
        if not os.path.isdir(sample_dir):
            continue
        for turn_name in sorted(os.listdir(sample_dir)):
            turn_dir = os.path.join(sample_dir, turn_name)
            metadata_path = os.path.join(turn_dir, "metadata.json")
            if os.path.exists(metadata_path):
                yield sample_dir, turn_dir, metadata_path


def parse_sample(sample_dir: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    match = SAMPLE_RE.match(os.path.basename(sample_dir))
    if not match:
        return None, None, None
    return match.group("index"), match.group("uid"), match.group("qhash")


def parse_turn(turn_dir: str) -> Optional[int]:
    match = TURN_RE.match(os.path.basename(turn_dir))
    if not match:
        return None
    return int(match.group("turn"))


def collect_records(root: str, max_snapshots: Optional[int]) -> Tuple[pd.DataFrame, Dict[Tuple[int, str], np.ndarray]]:
    summary = load_summary(root)
    records: List[Dict[str, Any]] = []
    vectors: Dict[Tuple[int, str], np.ndarray] = {}
    record_idx = 0

    for sample_dir, turn_dir, metadata_path in iter_metadata(root):
        sample_index, uid, qhash = parse_sample(sample_dir)
        turn = parse_turn(turn_dir)
        sample_summary = summary.get(os.path.abspath(sample_dir), {})
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        for snapshot in metadata.get("snapshots", []):
            tensor_path = snapshot.get("tensor_path")
            if not tensor_path or not os.path.exists(tensor_path):
                continue
            payload = torch.load(tensor_path, map_location="cpu")
            boundary = payload.get("boundary", snapshot.get("boundary"))
            for layer, tensor in payload.get("layers", {}).items():
                vector = tensor.float().numpy()
                key = (record_idx, str(layer))
                vectors[key] = vector
                records.append(
                    {
                        "record_id": record_idx,
                        "sample_index": sample_index,
                        "uid": uid,
                        "qhash": qhash,
                        "turn": turn,
                        "snapshot_index": snapshot.get("snapshot_index"),
                        "boundary": boundary,
                        "layer": str(layer),
                        "boundary_start_generated_token": snapshot.get("boundary_start_generated_token"),
                        "boundary_end_generated_token": snapshot.get("boundary_end_generated_token"),
                        "l2_norm": float(np.linalg.norm(vector)),
                        "mean": float(vector.mean()),
                        "std": float(vector.std()),
                        "max_abs": float(np.abs(vector).max()),
                        "question": sample_summary.get("question"),
                        "predicted_answer": sample_summary.get("predicted_answer"),
                        "num_turns": sample_summary.get("num_turns"),
                        "tensor_path": tensor_path,
                    }
                )
                record_idx += 1
                if max_snapshots is not None and record_idx >= max_snapshots:
                    return pd.DataFrame(records), vectors

    return pd.DataFrame(records), vectors


def add_step_metrics(df: pd.DataFrame, vectors: Dict[Tuple[int, str], np.ndarray]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["delta_l2_from_prev_snapshot"] = np.nan
    out["cosine_from_prev_snapshot"] = np.nan

    group_cols = ["sample_index", "uid", "layer"]
    for _, group in out.sort_values(["turn", "snapshot_index"]).groupby(group_cols, dropna=False):
        prev_vector = None
        for idx, row in group.iterrows():
            vector = vectors.get((int(row["record_id"]), str(row["layer"])))
            if vector is None:
                continue
            if prev_vector is not None:
                denom = np.linalg.norm(prev_vector) * np.linalg.norm(vector)
                out.at[idx, "delta_l2_from_prev_snapshot"] = float(np.linalg.norm(vector - prev_vector))
                out.at[idx, "cosine_from_prev_snapshot"] = float(np.dot(prev_vector, vector) / denom) if denom else np.nan
            prev_vector = vector
    return out


def plot_projection(
    df: pd.DataFrame,
    vectors: Dict[Tuple[int, str], np.ndarray],
    out_dir: str,
    method: str,
    layer: str,
    max_points: int,
) -> Optional[str]:
    layer_df = df[df["layer"] == str(layer)].copy()
    if layer_df.empty:
        return None
    if len(layer_df) > max_points:
        layer_df = layer_df.sample(max_points, random_state=42).sort_values("record_id")

    matrix = np.stack([vectors[(int(row.record_id), str(row.layer))] for row in layer_df.itertuples()])
    matrix = StandardScaler().fit_transform(matrix)

    if method == "pca":
        coords = PCA(n_components=2, random_state=42).fit_transform(matrix)
    elif method == "tsne":
        perplexity = min(30, max(5, (len(layer_df) - 1) // 3))
        coords = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(matrix)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    layer_df["x"] = coords[:, 0]
    layer_df["y"] = coords[:, 1]
    csv_path = os.path.join(out_dir, f"{method}_layer_{layer}.csv")
    png_path = os.path.join(out_dir, f"{method}_layer_{layer}.png")
    layer_df.to_csv(csv_path, index=False)

    boundaries = sorted(layer_df["boundary"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    colors = {boundary: cmap(i % 10) for i, boundary in enumerate(boundaries)}

    plt.figure(figsize=(9, 7))
    for boundary in boundaries:
        mask = layer_df["boundary"] == boundary
        plt.scatter(layer_df.loc[mask, "x"], layer_df.loc[mask, "y"], s=14, alpha=0.72, label=boundary, color=colors[boundary])

    for (_, uid), group in layer_df.groupby(["sample_index", "uid"], dropna=False):
        group = group.sort_values(["turn", "snapshot_index"])
        if len(group) > 1:
            plt.plot(group["x"], group["y"], color="0.70", linewidth=0.5, alpha=0.35)

    plt.title(f"{method.upper()} of boundary hidden states, layer {layer}")
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.legend(markerscale=1.5, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()
    return png_path


def write_group_stats(df: pd.DataFrame, out_dir: str) -> None:
    numeric_cols = ["l2_norm", "std", "max_abs", "delta_l2_from_prev_snapshot", "cosine_from_prev_snapshot"]
    group_stats = (
        df.groupby(["layer", "boundary"], dropna=False)[numeric_cols]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    group_stats.to_csv(os.path.join(out_dir, "boundary_layer_stats.csv"), index=False)

    turn_stats = (
        df.groupby(["layer", "turn", "boundary"], dropna=False)[numeric_cols]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    turn_stats.to_csv(os.path.join(out_dir, "turn_boundary_layer_stats.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SearchR1 latent boundary snapshots.")
    parser.add_argument("--latent_root", required=True, help="Directory containing summary.json and sample_*/turn_* snapshots.")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--layer", action="append", dest="layers", default=None)
    parser.add_argument("--max_snapshots", type=int, default=None)
    parser.add_argument("--max_projection_points", type=int, default=3000)
    parser.add_argument("--skip_tsne", action="store_true")
    args = parser.parse_args()

    latent_root = os.path.abspath(args.latent_root)
    out_dir = os.path.abspath(args.out_dir or os.path.join(latent_root, "analysis"))
    os.makedirs(out_dir, exist_ok=True)

    df, vectors = collect_records(latent_root, args.max_snapshots)
    df = add_step_metrics(df, vectors)
    index_path = os.path.join(out_dir, "latent_snapshot_index.csv")
    df.to_csv(index_path, index=False)

    if df.empty:
        print(json.dumps({"out_dir": out_dir, "num_records": 0}, indent=2))
        return

    write_group_stats(df, out_dir)
    layers = args.layers or sorted(df["layer"].unique(), key=lambda value: int(value))
    plots = []
    for layer in layers:
        pca_path = plot_projection(df, vectors, out_dir, "pca", str(layer), args.max_projection_points)
        if pca_path:
            plots.append(pca_path)
        if not args.skip_tsne:
            tsne_path = plot_projection(df, vectors, out_dir, "tsne", str(layer), args.max_projection_points)
            if tsne_path:
                plots.append(tsne_path)

    print(
        json.dumps(
            {
                "out_dir": out_dir,
                "num_records": int(len(df)),
                "num_samples": int(df[["sample_index", "uid"]].drop_duplicates().shape[0]),
                "index_csv": index_path,
                "plots": plots,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
