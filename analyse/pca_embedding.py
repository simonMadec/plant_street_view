#!/usr/bin/env python3
"""
PCA on AlphaEarth + TESSERA embeddings at in-situ points (GeoJSON).

Outputs PC1 vs PC2 colored by land-cover class; points must fall inside all rasters used.
CLI overrides paths in analyse/pca_embedding.py defaults.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    import rasterio
    from rasterio.errors import RasterioIOError
    from rasterio.warp import transform_bounds, transform as warp_transform
except ImportError:
    rasterio = None
    transform_bounds = None
    warp_transform = None

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# Default paths (overridable via CLI)
EMBEDDING_BASE = Path("/data/data2/plant_street_view/images_kika/embedding")
GEOJSON_PATH = Path(__file__).resolve().parent / "data/insitu/Terrain_OBSYDYA_Parakou_2025_Fusion2.geojson"
OUTPUT_FIGURE = Path(__file__).resolve().parent / "results/pca_embedding_classes.png"

# Main classes to include (set to None or empty to include all)
MAIN_CLASSES = [
    "Coton",
    "Autre Verger",
    "Manguier",
    "Sorgho/Mil",
    "Teck",
    "Mais",
    "Anacardier",
    "Soja",
]


def _check_deps() -> None:
    missing = []
    if gpd is None:
        missing.append("geopandas")
    if rasterio is None:
        missing.append("rasterio")
    if PCA is None:
        missing.append("scikit-learn")
    if plt is None:
        missing.append("matplotlib")
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}. Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def _discover_tessera_tiles(embedding_base: Path, year: int) -> list[tuple[Path, tuple[float, float, float, float]]]:
    """Return [(tiff_path, (left, bottom, right, top)), ...] for all TESSERA tiles of given year."""
    out: list[tuple[Path, tuple[float, float, float, float]]] = []
    if year == 2024:
        base = embedding_base / "tessera_embeddings" / "global_0.1_degree_representation" / "2024"
    else:
        base = embedding_base / "tessera_embeddings" / "2025" / "global_0.1_degree_representation" / "2025"
    if not base.exists():
        return out
    for grid_dir in sorted(base.iterdir()):
        if not grid_dir.is_dir() or not grid_dir.name.startswith("grid_"):
            continue
        tiff = grid_dir / f"{grid_dir.name}_{year}.tiff"
        if not tiff.exists():
            continue
        try:
            with rasterio.open(tiff) as src:
                b = src.bounds
                if src.crs and str(src.crs).lower() not in ("epsg:4326", "ogr:4326", "wgs84"):
                    l, bot, r, top = transform_bounds(src.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
                else:
                    l, bot, r, top = b.left, b.bottom, b.right, b.top
                out.append((tiff, (l, bot, r, top)))
        except Exception:
            continue
    return out


def _find_tile(
    lon: float,
    lat: float,
    tiles: list[tuple[Path, tuple[float, float, float, float]]],
) -> Path | None:
    """Return TESSERA tiff path containing (lon, lat), or None."""
    for tiff_path, (left, bottom, right, top) in tiles:
        if left <= lon < right and bottom <= lat < top:
            return tiff_path
    return None


def _sample_valid(src: "rasterio.DatasetReader", coord: tuple[float, float]) -> np.ndarray | None:
    """Sample raster at (lon, lat) in WGS84. Return vector or None if invalid."""
    try:
        vals = list(src.sample([coord]))[0]
        vec = np.array(vals, dtype=np.float64)
    except (IndexError, RasterioIOError, Exception):
        return None
    if np.any(np.isnan(vec)) or np.all(vec == 0):
        return None
    return vec


def _sample_valid_tessera(
    src: "rasterio.DatasetReader", lon: float, lat: float
) -> np.ndarray | None:
    """Sample TESSERA raster at (lon, lat) WGS84. Transforms to raster CRS then samples."""
    if src.crs is None or str(src.crs).lower() in ("epsg:4326", "ogr:4326", "wgs84"):
        return _sample_valid(src, (lon, lat))
    try:
        xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        coord = (float(xs[0]), float(ys[0]))
        return _sample_valid(src, coord)
    except Exception:
        return None


def main() -> int:
    _check_deps()

    ap = argparse.ArgumentParser(description="PCA on embeddings at in-situ points, plot by class.")
    ap.add_argument("--embedding-base", type=Path, default=EMBEDDING_BASE, help="Embedding root dir")
    ap.add_argument("--geojson", type=Path, default=GEOJSON_PATH, help="GeoJSON with points and class_name")
    ap.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FIGURE,
        help="Output figure base path; writes _alphaearth2024, _tessera2024, _tessera2025.png",
    )
    ap.add_argument("--pca-components", type=int, default=10, help="PCA n_components")
    args = ap.parse_args()

    embedding_base = args.embedding_base
    geojson_path = args.geojson
    output_path = args.output
    n_components = args.pca_components

    alphaearth_tif = embedding_base / "alphaearth" / "alphaearth_embeddings_2024.tif"
    if not alphaearth_tif.exists():
        print(f"Alphaearth TIF not found: {alphaearth_tif}")
        return 1
    if not geojson_path.exists():
        print(f"GeoJSON not found: {geojson_path}")
        return 1

    # Load GeoJSON, ensure WGS84
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is not None:
        try:
            if gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
        except Exception:
            gdf = gdf.to_crs(epsg=4326)
    if "class_name" not in gdf.columns:
        print("GeoJSON has no 'class_name' column.")
        return 1

    # Filter to main classes only
    if MAIN_CLASSES:
        gdf = gdf[gdf["class_name"].isin(MAIN_CLASSES)]
        if gdf.empty:
            print(f"No points found for classes: {MAIN_CLASSES}")
            return 1
        print(f"Filtered to {len(gdf)} points in main classes: {MAIN_CLASSES}")

    coords = [(float(g.x), float(g.y)) for g in gdf.geometry]
    class_names = gdf["class_name"].astype(str).tolist()

    # Discover TESSERA tiles (bounds-based lookup)
    tiles_2024 = _discover_tessera_tiles(embedding_base, 2024)
    tiles_2025 = _discover_tessera_tiles(embedding_base, 2025)
    if not tiles_2024 or not tiles_2025:
        print("No TESSERA 2024 or 2025 tiles found.")
        return 1

    # Points that fall in both TESSERA 2024 and 2025 (tile lookup only)
    candidates: list[tuple[float, float, str, Path, Path]] = []
    for (lon, lat), cname in zip(coords, class_names):
        p24 = _find_tile(lon, lat, tiles_2024)
        p25 = _find_tile(lon, lat, tiles_2025)
        if p24 is None or p25 is None:
            continue
        candidates.append((lon, lat, cname, p24, p25))

    # Alphaearth: open once, sample all candidates; keep only valid
    kept: list[tuple[float, float, str, Path, Path, np.ndarray]] = []
    with rasterio.open(alphaearth_tif) as src_ae:
        for lon, lat, cname, p24, p25 in candidates:
            ae = _sample_valid(src_ae, (lon, lat))
            if ae is None:
                continue
            kept.append((lon, lat, cname, p24, p25, ae))

    # Group by (tile_2024, tile_2025); open each pair once, batch-sample TESSERA
    groups: dict[tuple[Path, Path], list[tuple[float, float, str, np.ndarray]]] = defaultdict(list)
    for lon, lat, cname, p24, p25, ae in kept:
        groups[(p24, p25)].append((lon, lat, cname, ae))

    ae_rows: list[np.ndarray] = []
    t24_rows: list[np.ndarray] = []
    t25_rows: list[np.ndarray] = []
    labels: list[str] = []
    for (p24, p25), items in groups.items():
        with rasterio.open(p24) as src_24, rasterio.open(p25) as src_25:
            for lon, lat, cname, ae in items:
                t24 = _sample_valid_tessera(src_24, lon, lat)
                t25 = _sample_valid_tessera(src_25, lon, lat)
                if t24 is None or t25 is None:
                    continue
                ae_rows.append(ae)
                t24_rows.append(t24)
                t25_rows.append(t25)
                labels.append(cname)

    if not labels:
        print("No points fall inside all three embedding rasters. Nothing to plot.")
        return 1

    X_ae = np.vstack(ae_rows)
    X_t24 = np.vstack(t24_rows)
    X_t25 = np.vstack(t25_rows)
    X_all = np.hstack([X_ae, X_t24, X_t25])
    finite = np.isfinite(X_all).all(axis=1)
    if not np.any(finite):
        print("All sampled values contain inf/nan. Nothing to plot.")
        return 1
    X_ae = X_ae[finite]
    X_t24 = X_t24[finite]
    X_t25 = X_t25[finite]
    labels = [l for l, f in zip(labels, finite) if f]
    n_pts = X_ae.shape[0]
    print(f"Kept {n_pts} points inside all three rasters (finite only).")

    # Plotting setup: classes, colors, markers, output base
    classes = sorted(set(labels))
    # Use maximally distinct colors
    distinct_colors = [
        "#FF0000",  # red
        "#00AA00",  # green
        "#0000FF",  # blue
        "#FF8000",  # orange
        "#AA00AA",  # purple
        "#00DDDD",  # cyan
        "#AAAA00",  # yellow-green
        "#8B4513",  # brown
    ]
    # Use distinct markers
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "p", "h"]
    class_to_color = {c: distinct_colors[i % len(distinct_colors)] for i, c in enumerate(classes)}
    class_to_marker = {c: markers[i % len(markers)] for i, c in enumerate(classes)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_base = output_path.with_suffix("")

    def make_plot(X: np.ndarray, title: str, out_path: Path) -> None:
        n_comp = min(n_components, X.shape[0], X.shape[1])
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X)
        pc1, pc2 = X_pca[:, 0], X_pca[:, 1]
        ev1, ev2 = pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]
        print(f"  {title}: PC1 {ev1:.4f}, PC2 {ev2:.4f}")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        for c in classes:
            mask = np.array([l == c for l in labels], dtype=bool)
            ax.scatter(
                pc1[mask],
                pc2[mask],
                color=class_to_color[c],
                marker=class_to_marker[c],
                label=c,
                s=40,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.3,
            )
        ax.set_xlabel(f"PC1 ({ev1:.1%})")
        ax.set_ylabel(f"PC2 ({ev2:.1%})")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9, ncol=2)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    make_plot(
        X_ae,
        "PCA of alphaearth 2024 by class",
        out_base.parent / f"{out_base.name}_alphaearth2024.png",
    )
    make_plot(
        X_t24,
        "PCA of TESSERA 2024 by class",
        out_base.parent / f"{out_base.name}_tessera2024.png",
    )
    make_plot(
        X_t25,
        "PCA of TESSERA 2025 by class",
        out_base.parent / f"{out_base.name}_tessera2025.png",
    )
    print(
        f"Saved figures to {out_base.name}_alphaearth2024.png, "
        f"{out_base.name}_tessera2024.png, {out_base.name}_tessera2025.png"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
