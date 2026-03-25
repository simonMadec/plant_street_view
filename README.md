# plant_street_view

Scripts for Mapillary coverage, 360° panos → perspective crops, PlantNet IDs, optional depth (MoGe), Benin transect / Maxent, and a few QA plots. Paths in several files still point at `/data/data2/...`; adjust before running elsewhere.

## Setup

| Step | |
|------|---|
| Config | Copy `config.example.yaml` to `config.yaml` and set Mapillary + PlantNet keys. |
| Run location | Most scripts assume the working directory is this folder so `utils.*` imports resolve. |

## Main entry points

| Script | What it does |
|--------|----------------|
| `main.py` | Gridded Mapillary Graph API fetch → `mapillary_data_{site}.csv` → PlantNet on thumbnail URLs → CSV + Plotly map. Does not download full panos. |
| `reproject.py` | Equirect panos under `photosphere/` → left/right JPGs + projected metadata CSV (paths are hardcoded at top of file). |
| `create_projected_csv.py` | Rebuilds a projected-metadata CSV by scanning `photosphere/projected_views/` filenames. |
| `plantnet_parrallel.py` | PlantNet on **local** projected images; parallel requests + optional stats JSON. |
| `depth.py` | MoGe depth per image; writes `.npy` next to images / under `saved_depth_maps/`. |
| `visuresults.py` | Samples rows from a projected CSV and writes figure panels (calls `utils/visuresults.py`). |
| `process_benin_data_for_species_distribution.py` | Transect pipeline: offsets, thresholds, optional Maxent via elapid. Config JSON path: `--config` (default `process_benin_config.json`). |

## `utils/`

| Module | Role |
|--------|------|
| `util.py` | Mapillary fetch helper, PlantNet-by-URL (`main.py`), static map PNG. |
| `visuresults.py` | Parse `plantnet_data` from CSV and matplotlib figures. |
| `jeremy360.py` | Pano → left/right perspective (`py360convert`). |

## `analyse/`

| Script | Role |
|--------|------|
| `stat.py` | Score thresholds, species counts from a projected metadata CSV. |
| `compare_csvs.py` / `compare_projected_views.py` | Diff CSVs or two `projected_views` trees. |
| `check_projected_metadata.py` | Row/column/path checks on one CSV. |
| `depthstat.py` | Histograms over all depth `.npy` files in a folder. |
| `pca_embedding.py` | PCA on raster embeddings at in-situ points (GeoJSON + rasters). |

## Other

| Path | Note |
|------|------|
| `data/insitu/process.py` | Shapefile → GeoJSON with class labels from `nomenclature.csv`. |
| `notebook/` | Older notebooks; not required for the CLI scripts. |
| `old/` | Archived one-offs; ignore unless you need history. |

No single `requirements.txt` here—dependencies vary by script (e.g. `requests`, `pandas`, `torch`+MoGe for depth, `elapid`/`rasterio` for Benin). Install what you use.
