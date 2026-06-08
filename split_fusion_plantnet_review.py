"""
Split projected images into four buckets (fusion plot vs PlantNet) and optionally
write RGB + depth review figures with a global depth colormap scale. When DO_FULL_IMAGES
is True, the original projected JPG is copied into the same figures/{bucket}/ folder as
the *_review.png (same basename as the source image).

Edit the configuration block below; there is no CLI. CSV is under PROJECT_DIR; projected
images and depth maps live under IMAGE_DIR. Large outputs go to OUT_DIR/figures/ on the
data disk (review PNGs plus full JPGs when DO_FULL_IMAGES); there is no separate split/
copy step.
"""
from __future__ import annotations

import ast
import gc
import json
import shutil
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# =============================================================================
# Configuration (edit here — no command-line arguments)
# =============================================================================

# Fusion CSV (small; keep on project path).
PROJECT_DIR = Path("/home/simon/project/plant_street_view")
CSV_PATH = PROJECT_DIR / "data" / "images_enriched_052026.csv"

# Projected JPGs and MoGe depth .npy files (e.g. .../1495745381652933_right.jpg).
IMAGE_DIR = Path("/data/data2/plant_street_view/images_kika/projected_views")
DEPTH_DIR = IMAGE_DIR / "saved_depth_maps"

# Review PNGs and optional full JPGs — use data disk (same volume as images).
OUT_DIR = Path("/data/data2/plant_street_view/result/fusion_plantnet_split")

DO_FIGURES = True
# Copy the original projected JPG into figures/{bucket}/ next to *_review.png (detail view).
DO_FULL_IMAGES = True

# Class strings to ignore when deciding if fusion/plot labels are present (case-insensitive).
EXCLUDE_PLOT_CLASSES: list[str] = []

ROW_LIMIT: Optional[int] = None # or None 
DRY_RUN = False

PLANTNET_MIN_CONFIDENCE = 0.5
PLANTNET_TOP_SCORE_MIN = 0.15
PLANTNET_TOP_K = 5

# Global depth scale: percentiles over finite depth values across all figures in this run.
DEPTH_PERCENTILE_LOW = 1.0
DEPTH_PERCENTILE_HIGH = 99.0
# Loading every pixel from every .npy into one array can use tens of GB and get the process
# OOM-killed. We subsample each map before concatenating (fixed seed for reproducibility).
DEPTH_SAMPLE_PER_MAP = 512

BUCKETS = ("both", "only_plantnet", "only_plot", "nothing")


def _parse_plantnet_data(raw: Any) -> Optional[dict]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s.lower() == "nan":
            return None
        if s.lower().startswith("skip"):
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return None
    if isinstance(raw, dict):
        return raw
    return None


def _plot_class_counts(
    val: Any, excluded_lower: set[str]
) -> bool:
    """True if this column contributes a non-excluded plot label."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return False
    return s.lower() not in excluded_lower


def _fusion_plot_year_value(row: pd.Series, year: str) -> Any:
    """Prefer enriched matched_label_*; fall back to legacy matched_class_* for one year."""
    if year == "2024":
        keys = ("matched_label_2024", "matched_class_2024")
    elif year == "2025":
        keys = ("matched_label_2025", "matched_class_2025")
    else:
        raise ValueError("year must be '2024' or '2025'")
    for key in keys:
        v = row.get(key)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if not str(v).strip() or str(v).strip().lower() == "nan":
            continue
        return v
    return None


def row_has_plot_data(
    row: pd.Series, excluded_lower: set[str]
) -> bool:
    return _plot_class_counts(_fusion_plot_year_value(row, "2024"), excluded_lower) or _plot_class_counts(
        _fusion_plot_year_value(row, "2025"), excluded_lower
    )


def row_plantnet_high_confidence(row: pd.Series) -> bool:
    prob = row.get("best_match_probability")
    if prob is None or (isinstance(prob, float) and pd.isna(prob)):
        return False
    try:
        p = float(prob)
    except (TypeError, ValueError):
        return False
    if p <= PLANTNET_MIN_CONFIDENCE:
        return False
    data = _parse_plantnet_data(row.get("plantnet_data"))
    if data:
        results = data.get("results")
        if isinstance(results, list) and results:
            return True
    nm = row.get("best_match_scientific_name")
    if nm is None or (isinstance(nm, float) and pd.isna(nm)):
        return False
    return bool(str(nm).strip())


def assign_bucket(row: pd.Series, excluded_lower: set[str]) -> str:
    pn = row_plantnet_high_confidence(row)
    plot = row_has_plot_data(row, excluded_lower)
    if pn and plot:
        return "both"
    if pn and not plot:
        return "only_plantnet"
    if plot and not pn:
        return "only_plot"
    return "nothing"


def top_plantnet_hits(row: pd.Series) -> list[Tuple[str, float]]:
    hits: list[Tuple[str, float]] = []
    data = _parse_plantnet_data(row.get("plantnet_data"))
    if data:
        results = data.get("results", [])
        if not isinstance(results, list):
            return []
        for r in results:
            if not isinstance(r, dict):
                continue
            score = r.get("score")
            if score is None:
                continue
            try:
                sc = float(score)
            except (TypeError, ValueError):
                continue
            if sc <= PLANTNET_TOP_SCORE_MIN:
                continue
            sp = r.get("species") or {}
            if not isinstance(sp, dict):
                continue
            name = sp.get("scientificNameWithoutAuthor") or ""
            if not name:
                name = "(unknown)"
            hits.append((str(name), sc))
        hits.sort(key=lambda x: -x[1])
        return hits[:PLANTNET_TOP_K]

    raw_prob = row.get("best_match_probability")
    nm = row.get("best_match_scientific_name")
    if nm is None or (isinstance(nm, float) and pd.isna(nm)):
        return []
    name = str(nm).strip()
    if not name:
        return []
    if raw_prob is None or (isinstance(raw_prob, float) and pd.isna(raw_prob)):
        return []
    try:
        sc = float(raw_prob)
    except (TypeError, ValueError):
        return []
    if sc <= PLANTNET_TOP_SCORE_MIN:
        return []
    return [(name, sc)]


def load_depth_map(depth_dir: Path, stem: str) -> Optional[np.ndarray]:
    p = depth_dir / f"{stem}.npy"
    if not p.is_file():
        return None
    try:
        d = np.load(p)
    except Exception:
        return None
    if not isinstance(d, np.ndarray):
        return None
    return np.squeeze(d)


def collect_depth_values_for_percentile(
    depth_dir: Path,
    stems: list[str],
    sample_per_map: int,
    rng: np.random.Generator,
    *,
    desc: str = "Depth .npy (global scale)",
) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    for stem in tqdm(stems, desc=desc, unit="map"):
        d = load_depth_map(depth_dir, stem)
        if d is None:
            continue
        flat = d[np.isfinite(d)].ravel().astype(np.float32, copy=False)
        if flat.size == 0:
            continue
        n = min(sample_per_map, int(flat.size))
        if flat.size > n:
            idx = rng.choice(flat.size, size=n, replace=False)
            flat = flat[idx]
        chunks.append(flat)
    return chunks


def global_depth_vmin_vmax(
    depth_chunks: list[np.ndarray],
) -> Tuple[float, float]:
    if not depth_chunks:
        return 0.0, 1.0
    tqdm.write("Computing global depth percentiles (concatenating samples)...")
    all_vals = np.concatenate(depth_chunks)
    lo = float(np.percentile(all_vals, DEPTH_PERCENTILE_LOW))
    hi = float(np.percentile(all_vals, DEPTH_PERCENTILE_HIGH))
    if lo >= hi:
        hi = lo + 1e-6
    return lo, hi


def read_rgb(path: Path) -> Optional[np.ndarray]:
    try:
        import cv2

        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def render_review_figure(
    out_path: Path,
    rgb: Optional[np.ndarray],
    depth: Optional[np.ndarray],
    vmin: float,
    vmax: float,
    label_2024: str,
    label_2025: str,
    cone_label: str,
    plantnet_lines: list[str],
    title_suffix: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), gridspec_kw={"width_ratios": [1.1, 1.1, 0.75]})

    if rgb is not None:
        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
    else:
        axes[0].text(0.5, 0.5, "Image not found", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_title("RGB (missing)")
    axes[0].axis("off")

    if depth is not None:
        im = axes[1].imshow(depth, cmap="turbo", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Depth (m){title_suffix}")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].text(
            0.5,
            0.5,
            "No depth .npy",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title("Depth (missing)")
    axes[1].axis("off")

    text_lines = [
        f"2024: {label_2024}",
        f"2025: {label_2025}",
    ]
    if cone_label.strip() and cone_label != "—":
        text_lines.append(f"Cone 2024: {cone_label}")
    text_lines.append("")
    text_lines.append(f"PlantNet top (>{PLANTNET_TOP_SCORE_MIN}):")
    if plantnet_lines:
        text_lines.extend(f"  • {ln}" for ln in plantnet_lines)
    else:
        text_lines.append("  (none above threshold)")

    axes[2].text(
        0.02,
        0.98,
        "\n".join(text_lines),
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        transform=axes[2].transAxes,
    )
    axes[2].axis("off")
    axes[2].set_title("Labels")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fmt_label(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    s = str(val).strip()
    return s if s else "—"


def main() -> None:
    excluded_lower = {x.lower().strip() for x in EXCLUDE_PLOT_CLASSES if x.strip()}

    tqdm.write(f"Reading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    if ROW_LIMIT is not None:
        df = df.iloc[: int(ROW_LIMIT)].copy()

    tqdm.pandas(desc="Assign buckets")
    df["_bucket"] = df.progress_apply(lambda r: assign_bucket(r, excluded_lower), axis=1)

    counts = df["_bucket"].value_counts()
    tqdm.write("Bucket counts:")
    for b in BUCKETS:
        tqdm.write(f"  {b}: {int(counts.get(b, 0))}")

    fig_root = OUT_DIR / "figures"

    if (DO_FIGURES or DO_FULL_IMAGES) and not DRY_RUN:
        for b in BUCKETS:
            (fig_root / b).mkdir(parents=True, exist_ok=True)

    if not DO_FIGURES and not DO_FULL_IMAGES:
        return

    # Stems for depth range: all rows we will render (same rows as dataframe).
    stems: list[str] = []
    for _, row in df.iterrows():
        name = row.get("projected_file")
        if pd.isna(name) or not str(name).strip():
            continue
        stems.append(Path(str(name).strip()).stem)

    if DO_FIGURES:
        tqdm.write(
            "Sampling depth maps for global vmin/vmax — progress bar advances per .npy file loaded."
        )
        depth_rng = np.random.default_rng(42)
        depth_chunks = collect_depth_values_for_percentile(
            DEPTH_DIR, stems, DEPTH_SAMPLE_PER_MAP, depth_rng
        )
        vmin, vmax = global_depth_vmin_vmax(depth_chunks)
        del depth_chunks
        gc.collect()
        tqdm.write(
            f"Global depth scale (percentiles {DEPTH_PERCENTILE_LOW}-{DEPTH_PERCENTILE_HIGH}): vmin={vmin:.4f}, vmax={vmax:.4f}"
        )
    else:
        vmin, vmax = 0.0, 1.0

    fig_it = tqdm(
        df.iterrows(),
        total=len(df),
        desc="figures + full JPGs",
        unit="row",
        disable=DRY_RUN,
    )
    for _, row in fig_it:
        name = row.get("projected_file")
        if pd.isna(name) or not str(name).strip():
            continue
        fname = str(name).strip()
        stem = Path(fname).stem
        b = row["_bucket"]
        img_path = IMAGE_DIR / fname
        rgb = read_rgb(img_path) if img_path.is_file() else None
        depth = load_depth_map(DEPTH_DIR, stem)

        hits = top_plantnet_hits(row)
        plantnet_lines = [f"{n} ({s:.3f})" for n, s in hits]

        out_png = fig_root / b / f"{stem}_review.png"
        suffix = f"  [scale {vmin:.2f}–{vmax:.2f} m]"
        if DRY_RUN:
            tqdm.write(f"[dry-run] figure -> {out_png}")
            if DO_FULL_IMAGES:
                tqdm.write(f"[dry-run] full image -> {fig_root / b / fname}")
            continue

        if DO_FIGURES:
            render_review_figure(
                out_png,
                rgb,
                depth,
                vmin,
                vmax,
                fmt_label(_fusion_plot_year_value(row, "2024")),
                fmt_label(_fusion_plot_year_value(row, "2025")),
                fmt_label(row.get("matched_class_label_2024_cone")),
                plantnet_lines,
                suffix,
            )
            del rgb, depth

        if DO_FULL_IMAGES and img_path.is_file():
            dst_full = fig_root / b / fname
            shutil.copy2(img_path, dst_full)

    if DRY_RUN and (DO_FIGURES or DO_FULL_IMAGES):
        tqdm.write("[dry-run] skipped writing figures / full JPG copies")


if __name__ == "__main__":
    main()
