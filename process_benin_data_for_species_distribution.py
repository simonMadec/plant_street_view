#!/usr/bin/env python3
"""
Process Benin transect Kika data: PlantNet → GPS offsets → filters → Maxent (elapid).

Config: JSON via --config (default process_benin_config.json). Paths for CSVs/rasters
are in that file. See also process_benin_data_for_species_distribution.json.
Originated from 7_notebook_Benin_part2.ipynb.
"""

import pandas as pd
import numpy as np
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import glob
from datetime import datetime
import re

# Imports for advanced features (non-optional)
import xarray as xr
import elapid as ela
import geopandas as gpd
import rasterio
from rasterio import features
import plotly.express as px
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified) - create directory if needed
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # If file logging fails, just continue with console logging
            logger.warning(f"Could not set up file logging to {log_file}: {e}. Continuing with console logging only.")
    
    return logger


def extract_best_species_identify(d: Optional[dict]) -> pd.Series:
    """
    Extract the best species match from PlantNet JSON data.
    
    Args:
        d: PlantNet result dictionary or None
        
    Returns:
        Series with 'best_match_scientific_name' and 'best_match_probability'
    """
    if d is None or not isinstance(d, dict):
        return pd.Series({
            "best_match_scientific_name": None,
            "best_match_probability": np.nan
        })
    
    results = d.get("results", [])
    if not results:
        return pd.Series({
            "best_match_scientific_name": None,
            "best_match_probability": np.nan
        })
    
    # Sort results by score (descending) and get the best
    best = max(results, key=lambda r: r.get("score", float("-inf")))
    
    species_info = best.get("species", {})
    name = (
        species_info.get("scientificNameWithoutAuthor")
        or species_info.get("scientificName")
        or None
    )
    score = best.get("score", np.nan)

    return pd.Series({
        "best_match_scientific_name": name,
        "best_match_probability": score
    })


def offset_latlon(lat_deg: np.ndarray, lon_deg: np.ndarray, 
                  bearing_deg: np.ndarray, distance_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate offset latitude/longitude from a point given bearing and distance.
    
    Args:
        lat_deg: Latitude in degrees (numpy array)
        lon_deg: Longitude in degrees (numpy array)
        bearing_deg: Bearing in degrees (numpy array)
        distance_m: Distance in meters (scalar)
        
    Returns:
        Tuple of (lat_offset, lon_offset) in degrees
    """
    R = 6378137.0  # Earth radius in meters
    
    lat1 = np.deg2rad(lat_deg)
    lon1 = np.deg2rad(lon_deg)
    brng = np.deg2rad(bearing_deg)
    ang_dist = distance_m / R

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(ang_dist)
        + np.cos(lat1) * np.sin(ang_dist) * np.cos(brng)
    )

    lon2 = lon1 + np.arctan2(
        np.sin(brng) * np.sin(ang_dist) * np.cos(lat1),
        np.cos(ang_dist) - np.sin(lat1) * np.sin(lat2)
    )

    return np.rad2deg(lat2), np.rad2deg(lon2)


def calculate_lateral_offsets(data: pd.DataFrame, distance_m: float, 
                               logger: logging.Logger) -> pd.DataFrame:
    """
    Calculate lateral GPS offsets for left/right projected views.
    
    Args:
        data: DataFrame with 'lat', 'lon', 'compass_angle', 'projected_view' columns
        distance_m: Lateral offset distance in meters
        logger: Logger instance
        
    Returns:
        DataFrame with added 'lat_offset' and 'lon_offset' columns
    """
    logger.info(f"Calculating lateral offsets ({distance_m}m) for GPS points...")
    
    # Initialize offset columns (default to original coordinates)
    data = data.copy()
    data['lat_offset'] = data['lat']
    data['lon_offset'] = data['lon']
    
    # Masks for left/right views
    mask_right = data['projected_view'] == 'right'
    mask_left = data['projected_view'] == 'left'
    
    logger.debug(f"Found {mask_right.sum()} right views and {mask_left.sum()} left views")
    
    # RIGHT: compass_angle + 90°
    if mask_right.any():
        lat_r, lon_r = offset_latlon(
            data.loc[mask_right, 'lat'].values,
            data.loc[mask_right, 'lon'].values,
            data.loc[mask_right, 'compass_angle'].values + 90,
            distance_m
        )
        data.loc[mask_right, 'lat_offset'] = lat_r
        data.loc[mask_right, 'lon_offset'] = lon_r
    
    # LEFT: compass_angle - 90°
    if mask_left.any():
        lat_l, lon_l = offset_latlon(
            data.loc[mask_left, 'lat'].values,
            data.loc[mask_left, 'lon'].values,
            data.loc[mask_left, 'compass_angle'].values - 90,
            distance_m
        )
        data.loc[mask_left, 'lat_offset'] = lat_l
        data.loc[mask_left, 'lon_offset'] = lon_l
    
    logger.info("Lateral offsets calculated successfully")
    return data


def load_and_process_data(csv_path: str, distance_m: float, 
                          logger: logging.Logger) -> pd.DataFrame:
    """
    Load CSV data and process PlantNet results.
    
    Args:
        csv_path: Path to input CSV file
        distance_m: Lateral offset distance in meters
        logger: Logger instance
        
    Returns:
        Processed DataFrame with extracted species information and offsets
    """
    logger.info(f"Loading data from {csv_path}...")
    data = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(data)} rows")
    
    # Remove existing best_match_* columns (will recalculate)
    cols_to_drop = [c for c in data.columns if c.startswith("best_match_")]
    if cols_to_drop:
        logger.debug(f"Dropping columns: {cols_to_drop}")
        data = data.drop(columns=cols_to_drop)
    
    # Drop rows where plantnet_data is NaN
    initial_count = len(data)
    data = data.dropna(subset=["plantnet_data"])
    dropped_count = initial_count - len(data)
    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count} rows with missing plantnet_data")
    
    # Reset index
    data.reset_index(drop=True, inplace=True)
    
    # Parse plantnet_data JSON column
    logger.info("Parsing PlantNet JSON data...")
    def parse_json_safe(x):
        """Safely parse JSON string, return None if invalid."""
        if not isinstance(x, str) or not x.strip():
            return None
        try:
            return json.loads(x)
        except (json.JSONDecodeError, ValueError):
            return None
    
    data["plantnet_data_dict"] = data["plantnet_data"].apply(parse_json_safe)
    
    # Extract best match species
    logger.info("Extracting best species matches...")
    data[["best_match_scientific_name", "best_match_probability"]] = (
        data["plantnet_data_dict"].apply(extract_best_species_identify)
    )
    
    valid_matches = data["best_match_scientific_name"].notna().sum()
    logger.info(f"Found {valid_matches} valid species matches out of {len(data)} rows")
    
    # Calculate lateral offsets
    if 'compass_angle' in data.columns and 'projected_view' in data.columns:
        data = calculate_lateral_offsets(data, distance_m, logger)
    else:
        logger.warning("Missing 'compass_angle' or 'projected_view' columns. Skipping offset calculation.")
        data['lat_offset'] = data['lat']
        data['lon_offset'] = data['lon']
    
    return data


def filter_by_threshold(data: pd.DataFrame, plantnet_thres: float,
                        logger: logging.Logger) -> pd.DataFrame:
    """
    Filter data by PlantNet probability threshold.
    
    Args:
        data: DataFrame with 'best_match_probability' column
        plantnet_thres: Minimum probability threshold
        logger: Logger instance
        
    Returns:
        Filtered DataFrame
    """
    initial_count = len(data)
    filtered_data = data[data["best_match_probability"] >= plantnet_thres].copy()
    filtered_count = len(filtered_data)
    logger.info(f"Filtered data: {filtered_count}/{initial_count} rows above threshold {plantnet_thres}")
    
    return filtered_data


def create_run_output_dir(base_output_dir: str, logger: logging.Logger) -> Path:
    """
    Create a new run-specific output directory under the given base directory.
    
    Directory name pattern: YYYYMMDD_runNN (e.g., 20251223_run01).
    """
    base_dir = Path(base_output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    pattern = re.compile(rf"^{date_str}_run(\d+)$")

    existing_runs = []
    for child in base_dir.iterdir():
        if child.is_dir():
            m = pattern.match(child.name)
            if m:
                try:
                    existing_runs.append(int(m.group(1)))
                except ValueError:
                    continue

    next_run_number = (max(existing_runs) + 1) if existing_runs else 1
    run_dir_name = f"{date_str}_run{next_run_number:02d}"
    run_dir = base_dir / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=False)

    logger.info(f"Created run output directory: {run_dir}")
    return run_dir


def prepare_maxent_occurrences(filtered_data: pd.DataFrame, species_names: List[str],
                               logger: logging.Logger) -> gpd.GeoDataFrame:
    """
    Prepare occurrence data as GeoDataFrame for Elapid Maxent.
    
    Args:
        filtered_data: Filtered DataFrame with species detections
        species_names: List of species names to include
        logger: Logger instance
        
    Returns:
        GeoDataFrame with Point geometries and species column
    """
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required for Maxent processing. Install with: pip install geopandas")
    
    logger.info(f"Preparing Maxent occurrences for {len(species_names)} species...")
    
    # Filter to selected species
    occurrences = filtered_data[
        filtered_data['best_match_scientific_name'].isin(species_names)
    ].copy()
    
    logger.info(f"Found {len(occurrences)} occurrences for selected species")
    
    # Replace spaces with underscores in species names
    occurrences["species"] = occurrences["best_match_scientific_name"].str.replace(" ", "_")
    
    # Create Point geometries from lat/lon
    geometry = gpd.points_from_xy(occurrences['lon_offset'], occurrences['lat_offset'])
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        occurrences[["species"]],
        geometry=geometry,
        crs="EPSG:4326"
    )
    
    # Log species counts
    species_counts = gdf.groupby("species").size()
    logger.info("Species occurrence counts:")
    for species, count in species_counts.items():
        logger.info(f"  {species}: {count}")
    
    return gdf


def get_environmental_layer_paths(env_layers_dir: str, logger: logging.Logger) -> List[str]:
    """
    Get list of environmental raster file paths.
    
    Args:
        env_layers_dir: Directory containing environmental raster files
        logger: Logger instance
        
    Returns:
        List of raster file paths
    """
    env_dir = Path(env_layers_dir)
    if not env_dir.exists():
        logger.error(f"Environmental layers directory not found: {env_layers_dir}")
        return []
    
    # Find raster files (common extensions)
    raster_extensions = ['.tif', '.tiff', '.asc', '.geotiff']
    raster_files = []
    for ext in raster_extensions:
        raster_files.extend(list(env_dir.glob(f"*{ext}")))
        raster_files.extend(list(env_dir.glob(f"*{ext.upper()}")))
    
    raster_files = sorted(set(raster_files))  # Remove duplicates and sort
    
    if not raster_files:
        logger.error(f"No raster files found in {env_layers_dir}")
        return []
    
    logger.info(f"Found {len(raster_files)} environmental raster file(s)")
    
    # Count total bands/features
    total_bands = 0
    for rf in raster_files:
        try:
            with rasterio.open(rf) as src:
                n_bands = src.count
                total_bands += n_bands
                logger.info(f"  - {rf.name}: {n_bands} band(s)")
        except Exception as e:
            logger.warning(f"  - {rf.name}: Could not read bands ({e})")
    
    logger.info(f"Total number of features (bands) for Maxent: {total_bands}")
    
    return [str(f) for f in raster_files]


def run_maxent_elapid(occurrences_gdf: gpd.GeoDataFrame, env_layers_paths: List[str],
                      output_dir: str, maxent_thres: float, logger: logging.Logger,
                      n_background: int = 10000,
                      beta_lqp: float = 1.0,
                      beta_hinge: float = 1.0,
                      beta_threshold: float = 1.0,
                      beta_categorical: float = 1.0) -> Dict[str, str]:
    """
    Run Maxent using Elapid package.
    
    Args:
        occurrences_gdf: GeoDataFrame with Point geometries and 'species' column
        env_layers_paths: List of paths to environmental raster files
        output_dir: Directory to save Maxent predictions
        maxent_thres: Threshold for predictions (optional filtering)
        logger: Logger instance
        n_background: Number of background points to generate (default: 10000)
        beta_lqp: Regularization for linear, quadratic, and product features (default: 1.0)
        beta_hinge: Regularization for hinge features (default: 1.0)
        beta_threshold: Regularization for threshold features (default: 1.0)
        beta_categorical: Regularization for categorical features (default: 1.0)
        
    Returns:
        Dictionary mapping species names to output prediction file paths
    """
    if not HAS_ELAPID:
        raise ImportError("elapid is required. Install with: pip install elapid")
    
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required. Install with: pip install geopandas")
    
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required. Install with: pip install rasterio")
    
    logger.info("=" * 60)
    logger.info("Running Maxent using Elapid")
    logger.info("=" * 60)
    
    if not env_layers_paths:
        raise ValueError("No environmental layer paths provided")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique species
    species_list = occurrences_gdf['species'].unique()
    logger.info(f"Processing {len(species_list)} species: {list(species_list)}")
    
    # Count total features (bands) across all raster files
    total_features_count = 0
    for raster_path in env_layers_paths:
        with rasterio.open(raster_path) as src:
            total_features_count += src.count
            logger.info(f"  {Path(raster_path).name}: {src.count} band(s)")
    
    logger.info(f"Using {len(env_layers_paths)} raster file(s) with {total_features_count} total feature(s) for Maxent modeling")
    
    # Use first raster as template for output
    template_raster = env_layers_paths[0]
    
    output_files = {}
    all_metrics: List[Dict[str, float]] = []
    
    # Process each species separately
    for species in species_list:
        logger.info(f"\nProcessing species: {species}")
        
        # Filter occurrences for this species
        species_occurrences = occurrences_gdf[occurrences_gdf['species'] == species].copy()
        n_occurrences = len(species_occurrences)
        logger.info(f"  Occurrences: {n_occurrences}")
        
        if n_occurrences < 5:
            logger.warning(f"  Only {n_occurrences} occurrences for {species}. Skipping (need at least 5).")
            continue
        
        try:
            # Extract environmental values at occurrence points using rasterio
            logger.info(f"  Extracting environmental values at {n_occurrences} occurrence points...")
            coords = [(geom.x, geom.y) for geom in species_occurrences.geometry]
            
            # Initialize list to store all features for each sample
            X_presence = [[] for _ in range(len(coords))]
            total_features = 0
            
            for idx, raster_path in enumerate(env_layers_paths, 1):
                with rasterio.open(raster_path) as src:
                    n_bands = src.count
                    total_features += n_bands
                    logger.info(f"    Raster {idx}/{len(env_layers_paths)}: {Path(raster_path).name} - {n_bands} band(s)")
                    
                    # Extract all bands for each coordinate
                    for coord_idx, coord in enumerate(coords):
                        try:
                            # Get all bands for this point (returns tuple of values, one per band)
                            sample_values = list(src.sample([coord]))[0]
                            X_presence[coord_idx].extend(sample_values)
                        except (IndexError, rasterio.errors.RasterioIOError):
                            # Point outside raster, add NaN for all bands
                            X_presence[coord_idx].extend([np.nan] * n_bands)
            
            X_presence = np.array(X_presence)  # Shape: (n_samples, n_features)
            logger.info(f"  Total features extracted: {total_features} (from {len(env_layers_paths)} raster file(s))")
            logger.info(f"  Extracted presence data shape: {X_presence.shape} ({X_presence.shape[0]} samples x {X_presence.shape[1]} features)")
            
            # Generate background points (random sampling from raster extent)
            logger.info(f"  Generating {n_background} background points...")
            # Use first raster to get extent
            with rasterio.open(env_layers_paths[0]) as src:
                bounds = src.bounds
                # Generate random points within bounds
                x_coords = np.random.uniform(bounds.left, bounds.right, n_background)
                y_coords = np.random.uniform(bounds.bottom, bounds.top, n_background)
                background_coords = list(zip(x_coords, y_coords))
            
            logger.info(f"  Extracting values at {n_background} background points...")
            # Determine total feature count first
            n_features_per_sample = total_features
            
            X_background = []
            for coord_idx, coord in enumerate(background_coords):
                sample_values = []
                try:
                    for raster_path in env_layers_paths:
                        with rasterio.open(raster_path) as src:
                            # Extract all bands for this coordinate
                            band_values = list(src.sample([coord]))[0]
                            sample_values.extend(band_values)
                    X_background.append(sample_values)
                except (IndexError, rasterio.errors.RasterioIOError):
                    # Point outside raster or error, use NaN for all features
                    X_background.append([np.nan] * n_features_per_sample)
            
            X_background = np.array(X_background)
            # Remove rows with NaN
            n_before_filter = len(X_background)
            valid_mask = ~np.isnan(X_background).any(axis=1)
            X_background = X_background[valid_mask]
            n_after_filter = len(X_background)
            logger.info(f"  Background data shape: {X_background.shape} (samples x features), filtered {n_before_filter - n_after_filter} invalid points")
            
            # Combine presence and background
            X_combined = np.vstack([X_presence, X_background])
            y_combined = np.hstack([np.ones(len(X_presence)), np.zeros(len(X_background))])
            
            logger.info(f"  Training data: {len(X_presence)} presence, {len(X_background)} background")
            logger.info(f"  Combined training data shape: {X_combined.shape} ({X_combined.shape[0]} samples x {X_combined.shape[1]} features)")
            
            # Initialize and fit Maxent model
            logger.info(f"  Fitting Maxent model with {X_combined.shape[1]} feature(s)...")
            logger.info(f"  Regularization parameters: beta_lqp={beta_lqp}, beta_hinge={beta_hinge}, beta_threshold={beta_threshold}, beta_categorical={beta_categorical}")
            model = ela.MaxentModel(
                beta_lqp=beta_lqp,
                beta_hinge=beta_hinge,
                beta_threshold=beta_threshold,
                beta_categorical=beta_categorical
            )
            model.fit(X_combined, y_combined)
            logger.info(f"  Model fitted successfully")

            # Evaluate model performance on training data
            try:
                y_scores = model.predict(X_combined)
            except Exception as e:
                logger.warning(f"  Could not compute predictions for evaluation for {species}: {e}")
                y_scores = None

            metrics_row: Dict[str, float] = {
                "species": species,
                "n_presence": int(len(X_presence)),
                "n_background": int(len(X_background)),
                "n_samples": int(len(X_combined)),
            }

            if y_scores is not None:
                # Binarize using default threshold 0.5
                y_pred = (y_scores >= 0.5).astype(int)

                # Confusion matrix components
                tp = int(((y_combined == 1) & (y_pred == 1)).sum())
                tn = int(((y_combined == 0) & (y_pred == 0)).sum())
                fp = int(((y_combined == 0) & (y_pred == 1)).sum())
                fn = int(((y_combined == 1) & (y_pred == 0)).sum())

                total = tp + tn + fp + fn
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
                specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
                prevalence = (tp + fn) / total if total > 0 else float("nan")
                tss = (sensitivity + specificity - 1) if (
                    not np.isnan(sensitivity) and not np.isnan(specificity)
                ) else float("nan")

                metrics_row.update(
                    {
                        "tp": tp,
                        "tn": tn,
                        "fp": fp,
                        "fn": fn,
                        "sensitivity": sensitivity,
                        "specificity": specificity,
                        "tss": tss,
                        "prevalence": prevalence,
                    }
                )

                if HAS_SKLEARN:
                    try:
                        auc_roc = roc_auc_score(y_combined, y_scores)
                    except Exception:
                        auc_roc = float("nan")
                    try:
                        acc = accuracy_score(y_combined, y_pred)
                    except Exception:
                        acc = float("nan")
                    try:
                        prec = precision_score(y_combined, y_pred, zero_division=0)
                    except Exception:
                        prec = float("nan")
                    try:
                        rec = recall_score(y_combined, y_pred, zero_division=0)
                    except Exception:
                        rec = float("nan")
                    try:
                        f1 = f1_score(y_combined, y_pred, zero_division=0)
                    except Exception:
                        f1 = float("nan")

                    metrics_row.update(
                        {
                            "auc_roc": auc_roc,
                            "accuracy": acc,
                            "precision": prec,
                            "recall": rec,
                            "f1_score": f1,
                        }
                    )
                else:
                    logger.warning(
                        "scikit-learn not available. Install with 'pip install scikit-learn' "
                        "to enable additional Maxent evaluation metrics (AUC, accuracy, F1, etc.)."
                    )

            all_metrics.append(metrics_row)
            
            # Predict across entire study area
            logger.info("  Predicting habitat suitability across study area...")
            # For prediction, we need to read all rasters and all bands
            # Use the first raster as template
            with rasterio.open(env_layers_paths[0]) as src:
                profile = src.profile
                height, width = src.height, src.width
                transform = src.transform
                crs = src.crs
            
            # Read all environmental layers and all bands
            env_stacked = []
            logger.info(f"  Reading all bands from {len(env_layers_paths)} raster file(s) for prediction...")
            for raster_path in env_layers_paths:
                with rasterio.open(raster_path) as src:
                    n_bands = src.count
                    logger.debug(f"    Reading {n_bands} band(s) from {Path(raster_path).name}")
                    # Read all bands and flatten each
                    for band_idx in range(1, n_bands + 1):
                        data = src.read(band_idx)
                        env_stacked.append(data.flatten())
            
            env_stacked = np.array(env_stacked).T  # (n_pixels, n_features)
            logger.info(f"  Prediction data shape: {env_stacked.shape} ({env_stacked.shape[0]} pixels x {env_stacked.shape[1]} features)")
            
            # Predict
            predictions_flat = model.predict(env_stacked)
            predictions = predictions_flat.reshape(height, width)
            
            # Save prediction raster using rasterio
            output_filename = output_dir / f"{species}_prediction.tif"
            profile.update(dtype=rasterio.float32, count=1)
            with rasterio.open(output_filename, 'w', **profile) as dst:
                dst.write(predictions.astype(rasterio.float32), 1)
            
            output_files[species] = str(output_filename)
            logger.info(f"  Saved prediction to {output_filename}")
            
            # Create visualization
            viz_output = output_dir / f"{species}_prediction.png"
            create_maxent_prediction_visualization(str(output_filename), species, str(viz_output), logger)
            
            # Apply threshold if specified
            if maxent_thres > 0:
                thresholded_filename = output_dir / f"{species}_thresholded_{maxent_thres}.tif"
                predictions_thresh = np.where(predictions > maxent_thres, predictions, 0)
                profile.update(dtype=rasterio.float32, count=1)
                with rasterio.open(thresholded_filename, 'w', **profile) as dst:
                    dst.write(predictions_thresh.astype(rasterio.float32), 1)
                logger.info(f"  Saved thresholded prediction to {thresholded_filename}")
        
        except Exception as e:
            logger.error(f"  Error processing {species}: {e}", exc_info=True)
            continue
    
    # Save metrics for all species to CSV and JSON in the output directory
    if all_metrics:
        try:
            metrics_df = pd.DataFrame(all_metrics)
            metrics_csv_path = Path(output_dir) / "maxent_metrics.csv"
            metrics_json_path = Path(output_dir) / "maxent_metrics.json"
            metrics_df.to_csv(metrics_csv_path, index=False)
            metrics_df.to_json(metrics_json_path, orient="records", indent=2)
            logger.info(f"Saved Maxent metrics to {metrics_csv_path} and {metrics_json_path}")
        except Exception as e:
            logger.warning(f"Could not save Maxent metrics: {e}")

    logger.info("=" * 60)
    logger.info(f"Maxent processing completed. Generated {len(output_files)} prediction files.")
    logger.info("=" * 60)
    
    return output_files


def load_maxent_results(raster_files_pattern: str, maxent_thres: float,
                        logger: logging.Logger) -> Dict[str, str]:
    """
    Load Maxent result raster files (TIF format from Elapid).
    
    Args:
        raster_files_pattern: Glob pattern for Maxent output raster files (*_prediction.tif)
        maxent_thres: Threshold for Maxent probability values (for reference, already applied if thresholded files exist)
        logger: Logger instance
        
    Returns:
        Dictionary mapping species names to output prediction file paths
    """
    logger.info(f"Loading Maxent results from pattern: {raster_files_pattern}")
    
    # Handle invalid patterns (like "*" alone)
    if raster_files_pattern == "*" or not raster_files_pattern or raster_files_pattern.strip() == "*":
        logger.warning("Invalid raster_files_pattern '*'. Skipping Maxent results loading.")
        return {}
    
    raster_files = glob.glob(raster_files_pattern)
    logger.info(f"Found {len(raster_files)} raster files")
    
    result_files = {}
    for raster_file in raster_files:
        # Extract species name from filename (e.g., "Tectona_grandis_prediction.tif" -> "Tectona_grandis")
        filename = Path(raster_file).stem
        # Remove common suffixes
        for suffix in ['_prediction', '_thresholded']:
            if filename.endswith(suffix):
                filename = filename[:-len(suffix)]
                break
        species_name = filename
        
        result_files[species_name] = raster_file
        logger.debug(f"Found prediction for {species_name}: {raster_file}")
    
    logger.info(f"Successfully loaded {len(result_files)} species prediction files")
    
    return result_files


def create_maxent_prediction_visualization(prediction_raster_path: str, species_name: str,
                                           output_path: str, logger: logging.Logger) -> None:
    """
    Create PNG visualization of Maxent prediction raster.
    
    Args:
        prediction_raster_path: Path to prediction raster file
        species_name: Name of the species (for title)
        output_path: Path to save PNG visualization
        logger: Logger instance
    """
    if not HAS_MATPLOTLIB or not HAS_RASTERIO:
        logger.warning("matplotlib or rasterio not available. Skipping prediction visualization.")
        return
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        logger.info(f"  Creating visualization for {species_name}...")
        
        # Read raster
        with rasterio.open(prediction_raster_path) as src:
            prediction_data = src.read(1)
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            nodata = src.nodata
        
        # Handle nodata values
        if nodata is not None:
            prediction_data = np.where(prediction_data == nodata, np.nan, prediction_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate extent for proper geospatial display
        # imshow needs extent in (left, right, bottom, top) format
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        
        # Create visualization with colormap (viridis works well for probability/suitability)
        im = ax.imshow(prediction_data, cmap='viridis', interpolation='bilinear', 
                      extent=extent, aspect='auto', origin='upper')
        
        # Handle NaN values (set them to transparent or a specific color)
        im.set_clim(vmin=np.nanmin(prediction_data), vmax=np.nanmax(prediction_data))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Habitat Suitability (0-1)', rotation=270, labelpad=20, fontsize=12)
        
        # Set title
        species_display = species_name.replace("_", " ")
        ax.set_title(f'Maxent Habitat Suitability Prediction\n{species_display}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Format tick labels to show coordinates properly
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # Save figure
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"  Saved visualization to {output_path}")
        
    except Exception as e:
        logger.error(f"  Error creating visualization: {e}", exc_info=True)


def create_visualization(filtered_data: pd.DataFrame, output_path: Optional[str],
                         logger: logging.Logger, offset_distance_m: Optional[float] = None) -> None:
    """
    Create interactive map visualization of filtered detections.
    
    Args:
        filtered_data: Filtered DataFrame with detections
        output_path: Optional path to save HTML visualization
        logger: Logger instance
    """
    if not HAS_PLOTLY:
        logger.warning("plotly not available. Skipping visualization.")
        return
    
    logger.info("Creating interactive map visualization...")
    
    fig = px.scatter_mapbox(
        filtered_data,
        lat='lat_offset',
        lon='lon_offset',
        color='best_match_scientific_name',
        hover_data=['best_match_probability'],
        zoom=13,
        height=600
    )
    title_text = "Plant Detections with GPS Offsets"
    if offset_distance_m is not None:
        title_text += f" (offset = {offset_distance_m:.1f} m)"

    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        title=title_text,
        title_x=0.5,
    )
    
    if output_path:
        fig.write_html(output_path)
        logger.info(f"Visualization saved to {output_path}")
    else:
        fig.show()


def create_osm_visualization_png(filtered_data: pd.DataFrame, output_path: str,
                                  logger: logging.Logger, offset_distance_m: Optional[float] = None) -> None:
    """
    Create PNG visualization with OpenStreetMap background showing filtered detections.
    
    Args:
        filtered_data: Filtered DataFrame with detections (must have lat_offset, lon_offset columns)
        output_path: Path to save PNG visualization
        logger: Logger instance
    """
    if not HAS_GEOPANDAS:
        logger.warning("geopandas not available. Skipping OSM visualization.")
        return
    
    if not HAS_CONTEXTILY:
        logger.warning("contextily not available. Install with: pip install contextily")
        logger.warning("Skipping OSM visualization.")
        return
    
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available. Skipping OSM visualization.")
        return
    
    logger.info("Creating PNG visualization with OpenStreetMap background...")
    
    try:
        # Check required columns
        if 'lat_offset' not in filtered_data.columns or 'lon_offset' not in filtered_data.columns:
            logger.error("Required columns 'lat_offset' and 'lon_offset' not found in data.")
            return
        
        # Create GeoDataFrame from points
        gdf = gpd.GeoDataFrame(
            filtered_data,
            geometry=gpd.points_from_xy(filtered_data['lon_offset'], filtered_data['lat_offset']),
            crs='EPSG:4326'
        )
        
        # Reproject to Web Mercator (required for contextily)
        gdf_mercator = gdf.to_crs(epsg=3857)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot points colored by species
        species_list = gdf_mercator['best_match_scientific_name'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(species_list)))
        species_color_map = dict(zip(species_list, colors))
        
        for species in species_list:
            species_data = gdf_mercator[gdf_mercator['best_match_scientific_name'] == species]
            ax.scatter(
                species_data.geometry.x,
                species_data.geometry.y,
                c=[species_color_map[species]],
                label=species,
                alpha=0.6,
                s=30,
                edgecolors='black',
                linewidths=0.5
            )
        
        # Set extent with some padding
        bounds = gdf_mercator.total_bounds
        margin_x = (bounds[2] - bounds[0]) * 0.1
        margin_y = (bounds[3] - bounds[1]) * 0.1
        ax.set_xlim(bounds[0] - margin_x, bounds[2] + margin_x)
        ax.set_ylim(bounds[1] - margin_y, bounds[3] + margin_y)
        
        # Add OpenStreetMap tiles
        try:
            ctx.add_basemap(
                ax,
                source=ctx.providers.OpenStreetMap.Mapnik,
                crs=gdf_mercator.crs,
                attribution_size=8
            )
        except Exception as e:
            logger.warning(f"Could not add OpenStreetMap tiles: {e}")
            logger.warning("Continuing without basemap tiles...")
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=1)
        
        # Labels and title
        ax.set_xlabel('Longitude (Web Mercator)', fontsize=11)
        ax.set_ylabel('Latitude (Web Mercator)', fontsize=11)
        title = 'Plant Detections with GPS Offsets'
        if offset_distance_m is not None:
            title += f' (offset = {offset_distance_m:.1f} m)'
        title += '\n(OpenStreetMap Background)'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Remove axis ticks for cleaner look
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # Save figure
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path_obj, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"PNG visualization with OpenStreetMap saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating OSM visualization: {e}", exc_info=True)


def print_species_statistics(filtered_data: pd.DataFrame, top_n: int,
                             logger: logging.Logger) -> None:
    """
    Print statistics about detected species.
    
    Args:
        filtered_data: Filtered DataFrame with detections
        top_n: Number of top species to display
        logger: Logger instance
    """
    species_counts = filtered_data.groupby("best_match_scientific_name").size().sort_values(ascending=False)
    logger.info(f"\nTop {min(top_n, len(species_counts))} species by occurrence count:")
    for i, (species, count) in enumerate(species_counts.head(top_n).items(), 1):
        logger.info(f"  {i}. {species}: {count}")


def main(config_path: str):
    """Main processing function."""
    # Load config
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Setup logging
    log_level = config.get("logging", {}).get("level", "INFO")
    log_file = config.get("logging", {}).get("file")
    logger = setup_logging(log_level, log_file)
    
    logger.info("=" * 60)
    logger.info("Starting Benin transect data processing")
    logger.info("=" * 60)
    
    # Extract parameters
    input_csv = config["input"]["csv_path"]
    distance_m = config["processing"]["lateral_offset_distance_m"]
    plantnet_thres = config["processing"]["plantnet_threshold"]
    maxent_thres = config["processing"]["maxent_threshold"]
    
    # Load and process data
    data = load_and_process_data(input_csv, distance_m, logger)
    
    # Filter by threshold
    filtered_data = filter_by_threshold(data, plantnet_thres, logger)
    
    # Print statistics
    print_species_statistics(filtered_data, top_n=30, logger=logger)
    
    # Save filtered data
    output_csv = config["output"].get("filtered_csv_path")
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_data.to_csv(output_path, index=False)
        logger.info(f"Filtered data saved to {output_path}")
    
    # Run Maxent using Elapid (if configured)
    maxent_config = config.get("maxent", {})
    species_list = maxent_config.get("species_list")
    env_layers_dir = maxent_config.get("environmental_layers_dir")
    maxent_output_dir = maxent_config.get("output_dir")
    n_background = maxent_config.get("n_background", 10000)
    beta_lqp = maxent_config.get("beta_lqp", 1.0)
    beta_hinge = maxent_config.get("beta_hinge", 1.0)
    beta_threshold = maxent_config.get("beta_threshold", 1.0)
    beta_categorical = maxent_config.get("beta_categorical", 1.0)
    run_maxent = maxent_config.get("run_maxent", False)
    
    if run_maxent and species_list and env_layers_dir and maxent_output_dir:
        if not HAS_ELAPID:
            logger.error("Elapid not available. Install with: pip install elapid")
            logger.error("Skipping Maxent processing.")
        else:
            # Create a new run-specific output directory
            run_output_dir = create_run_output_dir(maxent_output_dir, logger)

            # Save processing parameters for this run
            run_params = {
                "timestamp": datetime.now().isoformat(),
                "config": config,
                "maxent": {
                    "species_list": species_list,
                    "environmental_layers_dir": env_layers_dir,
                    "base_output_dir": maxent_output_dir,
                    "run_output_dir": str(run_output_dir),
                    "n_background": n_background,
                    "beta_lqp": beta_lqp,
                    "beta_hinge": beta_hinge,
                    "beta_threshold": beta_threshold,
                    "beta_categorical": beta_categorical,
                    "maxent_threshold": maxent_thres,
                },
            }
            try:
                params_path = Path(run_output_dir) / "run_parameters.json"
                with open(params_path, "w") as pf:
                    json.dump(run_params, pf, indent=2)
                logger.info(f"Saved run parameters to {params_path}")
            except Exception as e:
                logger.warning(f"Could not save run parameters JSON: {e}")

            # Prepare occurrences as GeoDataFrame
            occurrences_gdf = prepare_maxent_occurrences(filtered_data, species_list, logger)
            
            # Get environmental layer paths
            env_layers_paths = get_environmental_layer_paths(env_layers_dir, logger)
            
            if env_layers_paths:
                # Run Maxent
                maxent_output_files = run_maxent_elapid(
                    occurrences_gdf, 
                    env_layers_paths,
                    str(run_output_dir),
                    maxent_thres,
                    logger,
                    n_background=n_background,
                    beta_lqp=beta_lqp,
                    beta_hinge=beta_hinge,
                    beta_threshold=beta_threshold,
                    beta_categorical=beta_categorical
                )
                logger.info(f"Maxent completed. Generated predictions for {len(maxent_output_files)} species.")
            else:
                logger.error("No environmental layers found. Skipping Maxent processing.")
    elif run_maxent:
        logger.warning("Maxent configuration incomplete. Need: species_list, environmental_layers_dir, output_dir")
    
    # Load Maxent results (if pattern provided - for existing results)
    maxent_raster_pattern = maxent_config.get("raster_files_pattern")
    if maxent_raster_pattern:
        result_files = load_maxent_results(maxent_raster_pattern, maxent_thres, logger)
        logger.info(f"Loaded {len(result_files)} existing Maxent result files")
    
    # Create visualization (if enabled)
    if config.get("output", {}).get("create_visualization", False):
        viz_path = config.get("output", {}).get("visualization_path")

        # Limit visualization to top 20 species (by occurrence count) to avoid clutter
        try:
            species_counts = (
                filtered_data.groupby("best_match_scientific_name")
                .size()
                .sort_values(ascending=False)
            )
            top_n = 20
            top_species = species_counts.head(top_n).index.tolist()
            filtered_for_viz = filtered_data[
                filtered_data["best_match_scientific_name"].isin(top_species)
            ].copy()
            logger.info(
                f"Creating visualizations using top {len(top_species)} species by occurrence count."
            )
        except Exception as e:
            logger.warning(
                f"Could not determine top species for visualization, falling back to all species: {e}"
            )
            filtered_for_viz = filtered_data

        create_visualization(filtered_for_viz, viz_path, logger, offset_distance_m=distance_m)
        
        # Also create PNG visualization with OpenStreetMap background
        if viz_path:
            png_path = str(Path(viz_path).with_suffix('.png'))
            create_osm_visualization_png(filtered_for_viz, png_path, logger, offset_distance_m=distance_m)
    
    logger.info("=" * 60)
    logger.info("Processing completed successfully")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process Benin transect Kika data: Extract PlantNet results, "
                    "calculate GPS offsets, filter detections, and process Maxent results."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="process_benin_config.json",
        help="Path to JSON configuration file (default: process_benin_config.json)"
    )
    
    args = parser.parse_args()
    
    try:
        main(args.config)
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
