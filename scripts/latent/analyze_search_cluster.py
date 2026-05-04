"""
Analyze the anomalous central cluster in </search> boundary hidden states.

Steps:
  1. Collect all search_query_end snapshots from summary.json
  2. PCA → 2D
  3. Histogram valley on PCA2 → threshold → central vs outer
  4. HTML report with search queries per cluster

Usage:
    python analyze_search_cluster.py \
        --latent_root /path/to/0427/XXX_searchr1_generate_latent \
        --out_dir     /path/to/analysis_search_cluster \
        [--layer 28]
"""

import argparse
import html as htmllib
import json
import os
import re

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SEARCH_END_BOUNDARIES = {"search_query_end", "</search>"}
SKIP_PREFIXES = ("think_dense_token_", "think_token_")
SKIP_EXACT = {"pre_generation", "think_first_content"}


# ── helpers ───────────────────────────────────────────────────────────────────

def extract_search_queries(full_response):
    if not full_response:
        return []
    queries = re.findall(r"<search>(.*?)</search>", full_response, re.DOTALL | re.IGNORECASE)
    return [q.strip() for q in queries]


def repetition_ratio(queries):
    if len(queries) <= 1:
        return 0.0
    unique = len(set(q.strip().lower() for q in queries))
    return round(1.0 - unique / len(queries), 2)


# ── data loading ──────────────────────────────────────────────────────────────

def collect_search_snapshots(latent_root, layer):
    """
    Collect hidden state vectors + metadata for all search_query_end snapshots.
    Returns (vectors list[np.ndarray], meta list[dict]).
    """
    with open(os.path.join(latent_root, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    vectors = []
    meta = []

    for row in summary:
        idx = row.get("index")
        question = row.get("question", "")
        search_queries = extract_search_queries(row.get("full_response", ""))

        for turn_info in row.get("latent_trace", []):
            turn = turn_info.get("turn")
            for snap in turn_info.get("snapshots", []):
                boundary = snap.get("boundary", "")
                if boundary not in SEARCH_END_BOUNDARIES:
                    continue
                tensor_path = snap.get("tensor_path")
                if not tensor_path or not os.path.exists(tensor_path):
                    continue
                try:
                    payload = torch.load(tensor_path, map_location="cpu")
                except Exception:
                    continue
                layers = payload.get("layers", {})
                key = str(layer)
                if key not in layers:
                    continue

                vec = layers[key].float().numpy()
                vectors.append(vec)
                meta.append({
                    "index": idx,
                    "turn": turn,
                    "boundary": boundary,
                    "question": question,
                    "search_queries": search_queries,
                })

    return vectors, meta


# ── threshold via histogram valley ────────────────────────────────────────────

def find_valley_threshold(pca2_values, out_path=None):
    vals = np.array(pca2_values)
    bins = 60
    counts, edges = np.histogram(vals, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    smoothed = gaussian_filter1d(counts.astype(float), sigma=2)
    peaks, _ = find_peaks(smoothed, height=smoothed.max() * 0.1, distance=5)

    threshold = None
    if len(peaks) >= 2:
        top2 = peaks[np.argsort(smoothed[peaks])[-2:]]
        top2 = sorted(top2)
        valley_region = smoothed[top2[0]:top2[1] + 1]
        valley_idx = top2[0] + int(np.argmin(valley_region))
        threshold = float(centers[valley_idx])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9, alpha=0.5, color="#4C78A8", label="count")
    ax.plot(centers, smoothed, color="#E45756", linewidth=1.5, label="smoothed")
    for p in peaks:
        ax.axvline(centers[p], color="green", linestyle="--", alpha=0.7, linewidth=1)
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="-", linewidth=2,
                   label=f"threshold={threshold:.2f}")
    ax.set_xlabel("PCA2")
    ax.set_ylabel("count")
    ax.set_title("PCA2 distribution of </search> snapshots")
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")
    return threshold


# ── scatter plot ──────────────────────────────────────────────────────────────

def plot_scatter(coords, cluster_labels, threshold, layer, out_path):
    central = coords[np.array(cluster_labels) == "central"]
    outer   = coords[np.array(cluster_labels) == "outer"]

    fig, ax = plt.subplots(figsize=(9, 7))
    if len(outer):
        ax.scatter(outer[:, 0], outer[:, 1], s=18, alpha=0.55, color="#4C78A8",
                   label=f"outer (PCA2 < {threshold:.1f}, n={len(outer)})")
    if len(central):
        ax.scatter(central[:, 0], central[:, 1], s=18, alpha=0.7, color="#E45756",
                   label=f"central (PCA2 ≥ {threshold:.1f}, n={len(central)})")
    if threshold is not None:
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1.2,
                   label=f"threshold PCA2={threshold:.2f}")
    ax.set_title(f"PCA of </search> hidden states — layer {layer}\ncentral vs outer by PCA2 valley", fontsize=11)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()
    print(f"Saved: {out_path}")


# ── HTML report ───────────────────────────────────────────────────────────────

def write_html(central_samples, outer_samples, hist_img, scatter_img, out_path, layer, dataset_name, threshold):
    def render_samples(samples, title, color, max_n=100):
        parts = [f"<h2 style='color:{color}'>{htmllib.escape(title)} (n={len(samples)} unique samples)</h2>"]
        for s in samples[:max_n]:
            rep = repetition_ratio(s["search_queries"])
            rep_style = "color:#c62828;font-weight:bold" if rep >= 0.4 else ""
            parts.append("<div class='sample'>")
            parts.append(f"<p class='q'><b>[idx={s['index']}]</b> {htmllib.escape(s['question'])}</p>")
            parts.append("<table><tr><th>#</th><th>search query</th></tr>")
            prev = None
            for i, q in enumerate(s["search_queries"]):
                dup = (q.strip().lower() == (prev or "").strip().lower())
                style = "background:#fff3e0" if dup else ""
                parts.append(f"<tr style='{style}'><td>{i}</td><td>{htmllib.escape(q)}</td></tr>")
                prev = q
            parts.append("</table>")
            parts.append(f"<p style='font-size:12px;{rep_style}'>repetition ratio: {rep}</p>")
            parts.append("</div>")
        return "\n".join(parts)

    hist_name    = os.path.basename(hist_img)
    scatter_name = os.path.basename(scatter_img)

    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Search Cluster — {htmllib.escape(dataset_name)} layer {layer}</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;color:#222;line-height:1.45}}
h1{{border-bottom:2px solid #ccc;padding-bottom:8px}}
.imgs{{display:flex;gap:20px;flex-wrap:wrap;margin:16px 0}}
.imgs img{{max-width:640px;border:1px solid #ccc}}
.stats{{background:#e8f5e9;padding:10px 16px;border-radius:4px;font-size:13px;margin:12px 0}}
.sample{{border:1px solid #ddd;padding:12px 16px;margin:8px 0;border-radius:4px;background:#fafafa}}
.q{{font-size:14px;margin:0 0 8px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:4px}}
td,th{{border:1px solid #ddd;padding:3px 8px;text-align:left}}
th{{background:#f0f0f0}}
</style></head><body>
<h1>Search Cluster Analysis — {htmllib.escape(dataset_name)}, layer {layer}</h1>
<p>All <code>&lt;/search&gt;</code> boundary snapshots projected to PCA 2D.<br>
Separation: <b>PCA2 threshold = {threshold:.2f}</b> (valley between two histogram peaks).</p>

<div class='stats'>
  PCA2 ≥ {threshold:.2f} → <b style='color:#c62828'>Central cluster</b> (anomalous — likely search stagnation / repeated queries)<br>
  PCA2 &lt; {threshold:.2f} → <b style='color:#1565c0'>Outer cluster</b> (normal distribution)
</div>

<div class='imgs'>
  <div><p><b>PCA2 histogram with valley threshold</b></p><img src='{htmllib.escape(hist_name)}'></div>
  <div><p><b>PCA scatter colored by cluster</b></p><img src='{htmllib.escape(scatter_name)}'></div>
</div>

{render_samples(central_samples, "Central Cluster (PCA2 ≥ threshold)", "#c62828")}
{render_samples(outer_samples,   "Outer Cluster (PCA2 < threshold)",   "#1565c0")}
</body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--layer", default="28")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    layer = args.layer

    print(f"Collecting </search> snapshots (layer {layer})...")
    vectors, meta = collect_search_snapshots(args.latent_root, layer)
    print(f"  {len(vectors)} snapshots collected")

    if len(vectors) < 4:
        print("Too few snapshots, aborting.")
        return

    # PCA
    matrix = np.stack(vectors).astype(np.float32)
    matrix_scaled = StandardScaler().fit_transform(matrix)
    coords = PCA(n_components=2, random_state=42).fit_transform(matrix_scaled)
    print(f"  PCA done: {coords.shape}")

    pca2_vals = coords[:, 1].tolist()

    # histogram valley
    hist_path = os.path.join(args.out_dir, f"pca2_histogram_layer_{layer}.png")
    threshold = find_valley_threshold(pca2_vals, out_path=hist_path)
    if threshold is None:
        print("WARNING: no clear valley found, using median as fallback.")
        threshold = float(np.median(pca2_vals))
    print(f"  Threshold PCA2 = {threshold:.3f}")

    cluster_labels = np.where(coords[:, 1] >= threshold, "central", "outer")

    # scatter
    scatter_path = os.path.join(args.out_dir, f"pca_scatter_layer_{layer}.png")
    plot_scatter(coords, cluster_labels, threshold, layer, scatter_path)

    # unique samples per cluster
    def unique_samples(label):
        seen = {}
        for m, l in zip(meta, cluster_labels):
            if l != label:
                continue
            idx = m["index"]
            if idx not in seen:
                seen[idx] = m
        return list(seen.values())

    central_samples = unique_samples("central")
    outer_samples   = unique_samples("outer")
    print(f"  Central: {len(central_samples)} unique samples")
    print(f"  Outer:   {len(outer_samples)} unique samples")

    # HTML
    dataset_name = os.path.basename(args.latent_root)
    html_path = os.path.join(args.out_dir, f"search_cluster_layer_{layer}.html")
    write_html(central_samples, outer_samples,
               hist_path, scatter_path,
               html_path, layer, dataset_name, threshold)

    print("Done.")


if __name__ == "__main__":
    main()
