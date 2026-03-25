#!/usr/bin/env python3
"""
Process in-situ shapefile: map Class codes to names via nomenclature.csv,
output GeoJSON of points with class (and code).
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


def load_nomenclature(path: Path) -> dict[int, str]:
    """Load nomenclature CSV (no header): code -> class_name. Skip empty lines."""
    mapping: dict[int, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or len(row) < 2:
                continue
            code_str, name = row[0].strip(), row[1].strip()
            if not code_str:
                continue
            try:
                mapping[int(code_str)] = name
            except ValueError:
                continue
    return mapping


def main() -> None:
    if not HAS_GEOPANDAS:
        sys.exit("geopandas is required. Install with: pip install geopandas")

    default_dir = Path(__file__).resolve().parent
    default_shape = default_dir / "Terrain_OBSYDYA_Parakou_2025_Fusion2.shp"
    default_nom = default_dir / "nomenclature.csv"

    parser = argparse.ArgumentParser(
        description="Convert in-situ shapefile to GeoJSON with points and class."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=default_shape,
        help=f"Path to shapefile (default: {default_shape})",
    )
    parser.add_argument(
        "--nomenclature",
        "-n",
        type=Path,
        default=default_nom,
        help=f"Path to nomenclature CSV (default: {default_nom})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output GeoJSON path (default: same dir/base as shapefile, .geojson)",
    )
    args = parser.parse_args()

    shp_path = args.input
    if shp_path.suffix.lower() != ".shp":
        shp_path = Path(str(shp_path) + ".shp")

    if not shp_path.exists():
        sys.exit(f"Shapefile not found: {shp_path}")

    if not args.nomenclature.exists():
        sys.exit(f"Nomenclature file not found: {args.nomenclature}")

    nomencl = load_nomenclature(args.nomenclature)
    if not nomencl:
        sys.exit("Nomenclature file is empty or has no valid code,name rows.")

    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        sys.exit("Shapefile is empty.")

    if "Class" not in gdf.columns:
        sys.exit("Shapefile has no 'Class' column.")

    def safe_int(v) -> int | None:
        if v is None or (hasattr(v, "__float__") and str(v) == "nan"):
            return None
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return None

    def class_name(code: int | None) -> str:
        if code is None:
            return "Unknown"
        return nomencl.get(code, "Unknown")

    codes = gdf["Class"].map(safe_int)
    gdf = gdf.assign(code=codes, class_name=codes.map(class_name))

    out_cols = ["geometry", "code", "class_name"]
    gdf_out = gdf[out_cols].copy()
    gdf_out = gdf_out.to_crs(epsg=4326)

    out_path = args.output
    if out_path is None:
        out_path = shp_path.with_suffix(".geojson")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_out.to_file(out_path, driver="GeoJSON")

    print(f"Wrote {len(gdf_out)} points to {out_path}")


if __name__ == "__main__":
    main()
