"""
Parse PlantNet JSON from CSV and render QA figures (random / crop-filtered samples).

Imported by visuresults.py; not used by main.py’s URL-based pipeline.
"""
import pandas as pd
import matplotlib.pyplot as plt
import json
import cv2
from pathlib import Path
import random
import numpy as np
import ast

def has_valid_data(x):
    """Check if the plantnet_data column has valid data."""
    if pd.isna(x) or x is None:
        return False
    if isinstance(x, str) and (x.strip() == "" or x.lower() == "nan" or x.lower() == "null"):
        return False
    return True

def get_image_id(filename):
    """Extract image ID from filename (remove _left/_right suffix)."""
    return str(filename).replace('.jpg', '').split('_')[0]

def parse_plantnet_data(plantnet_data_str):
    """Parses PlantNet data string into a dictionary."""
    try:
        if isinstance(plantnet_data_str, str):
            try:
                return json.loads(plantnet_data_str)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(plantnet_data_str)
                except:
                    return None
        else:
            return plantnet_data_str
    except Exception:
        return None

def get_plantnet_results(data):
    """Extracts list of (name, score, common_name) from parsed PlantNet data."""
    results = []
    if isinstance(data, dict):
        # Handle new API structure
        if 'results' in data and isinstance(data['results'], list):
            for res in data['results']: 
                score = res.get('score', 0)
                species = res.get('species', {})
                name = species.get('scientificNameWithoutAuthor', '')
                if not name:
                    name = f"{species.get('genus', '')} {species.get('species', '')}"
                
                # Extract common name
                common_names = species.get('commonNames', []) or species.get('vernacularNames', [])
                if not common_names:
                    common_names = species.get('gbif', {}).get('vernacularNames', [])
                common_name = common_names[0] if common_names and isinstance(common_names, list) and len(common_names) > 0 else ''
                
                results.append((name, score, common_name))
        
        # Handle old structure if any
        elif 'results' in data and isinstance(data['results'], dict): # Old structure
             species_list = data['results'].get('species', [])
             for s in species_list:
                 score = s.get('score', 0) # or max_score
                 name = s.get('scientificNameWithoutAuthor', s.get('name', 'Unknown'))
                 common_name = s.get('commonName', '') or s.get('vernacularName', '')
                 results.append((name, score, common_name))
    return results

def has_high_score(plantnet_data_str, threshold):
    """Checks if any result has score > threshold."""
    data = parse_plantnet_data(plantnet_data_str)
    if not data:
        return False
    
    results = get_plantnet_results(data)
    for _, score, _ in results:
        if score > threshold:
            return True
    return False

def visualize_plantnet_samples(
    csv_path,
    images_dir,
    output_dir,
    num_samples=5,
    seed=None,
    crop_name=None,
    score_threshold=0.0,
    depth=False,
    target_id=None
):
    """
    Visualizes random samples of PlantNet identifications from a CSV file.

    Args:
        csv_path (str or Path): Path to the CSV file containing metadata and plantnet_data.
        images_dir (str or Path): Directory containing the projected images.
        output_dir (str or Path): Directory where visualization images will be saved.
        num_samples (int): Number of images to visualize.
        seed (int, optional): Random seed for reproducibility.
        crop_name (str, optional): If provided, filters for images containing this crop name.
        score_threshold (float, optional): Minimum score threshold for crop detection or display.
        depth (bool, optional): If True, displays depth map alongside the image.
        target_id (str or int, optional): If provided, selects only images with this specific ID (e.g. "12345" for "12345_left.jpg").
    """
    if seed is not None:
        random.seed(seed)

    csv_path = Path(csv_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    if crop_name:
        output_dir = output_dir / crop_name
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    print(f"Reading CSV: {csv_path}")
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Create image_id column for all rows
    df['image_id'] = df['projected_file'].apply(get_image_id)

    # Base pool for sampling
    df_filtered = df.copy()

    # Apply content-based filters ONLY if specific criteria (score/crop) are requested
    if score_threshold > 0.0 or crop_name:
        # Filter rows that have valid plantnet_data first
        df_filtered = df_filtered[df_filtered['plantnet_data'].apply(has_valid_data)].copy()
        print(f"Found {len(df_filtered)} images with valid PlantNet data")

        if len(df_filtered) == 0:
            print("No valid data to filter!")
            return

        # Filter by score threshold
        if score_threshold > 0.0:
            print(f"Filtering for images with at least one score > {score_threshold}...")
            df_filtered = df_filtered[df_filtered['plantnet_data'].apply(
                lambda x: has_high_score(x, score_threshold)
            )].copy()
            print(f"Found {len(df_filtered)} images matching score criteria")
            
            if len(df_filtered) == 0:
                print(f"No images found with score > {score_threshold}")
                return

        # Filter by crop if specified
        if crop_name:
            print(f"Filtering for crop '{crop_name}'...")
            # We re-use the check logic but specific to crop now
            def check_crop_specific(x, c_name, thresh):
                 data = parse_plantnet_data(x)
                 if not data: return False
                 for name, score, common_name in get_plantnet_results(data):
                     if score > thresh and (c_name.lower() in name.lower() or c_name.lower() in common_name.lower()):
                         return True
                 return False

            df_filtered = df_filtered[df_filtered['plantnet_data'].apply(
                lambda x: check_crop_specific(x, crop_name, score_threshold)
            )].copy()
            print(f"Found {len(df_filtered)} images matching crop criteria")
            
            if len(df_filtered) == 0:
                print(f"No images found containing '{crop_name}' with score > {score_threshold}")
                return

    # Filter by Target ID if provided (applies to whatever pool we have)
    if target_id is not None:
        target_id_str = str(target_id)
        print(f"Selecting specific Image ID: {target_id_str}")
        df_filtered = df_filtered[df_filtered['image_id'] == target_id_str].copy()
        print(f"Found {len(df_filtered)} images for ID {target_id_str}")
        
        if len(df_filtered) == 0:
            print(f"No images found for ID {target_id_str}")
            return

    # Selection logic
    if target_id is not None:
        # If specific ID requested, show all matching rows (left/right) that survived filters
        samples = df_filtered
    elif crop_name or score_threshold > 0.0:
        # If searching for a crop OR using a threshold, we want individual valid images
        available_indices = df_filtered.index.tolist()
        actual_num_samples = min(num_samples, len(available_indices))
        if actual_num_samples == 0:
            samples = pd.DataFrame()
        else:
            sampled_indices = random.sample(available_indices, actual_num_samples)
            samples = df_filtered.loc[sampled_indices]
    else:
        # Original logic: Select random locations (unique IDs) to show pairs
        # This now includes locations without data if no filters were set
        unique_ids = df_filtered['image_id'].unique()
        print(f"Found {len(unique_ids)} unique locations in pool")
        
        actual_num_samples = min(num_samples, len(unique_ids))
        if actual_num_samples == 0:
            samples = pd.DataFrame()
        else:
            sampled_ids = random.sample(list(unique_ids), actual_num_samples)
            
            samples = df_filtered[df_filtered['image_id'].isin(sampled_ids)]
            samples = samples.sort_values('image_id')

    for idx, row in samples.iterrows():
        image_name = row['projected_file']
        image_path = images_dir / image_name
        plantnet_data_str = row['plantnet_data']
        
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue
            
        data = parse_plantnet_data(plantnet_data_str)
        all_results = []
        if data:
            all_results = get_plantnet_results(data)

        # Filter results for display based on threshold
        display_results = []
        for name, score, common_name in all_results:
            if score > score_threshold: # Use the threshold for filtering display too
                display_results.append((name, score, common_name))
        
        # Show ALL results (not just top 5)

        if (score_threshold > 0 or crop_name) and not display_results:
            print(f"No identifications > {score_threshold} for {image_name}, skipping")
            continue

        # Plot
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # We will use GridSpec to have fine control over layout
        # We want the text to be a small strip at the bottom
        
        # Plot
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # We attach text directly to the axes to ensure it is close to the image
        # regardless of aspect ratio issues.
        
        if depth:
            # Layout: 2 rows, 2 columns
            # Row 0: Image (Left), Depth (Right)
            # Row 1: Text (Left), Histogram (Right)
            plt.figure(figsize=(20, 10)) 
            gs = plt.GridSpec(2, 2, height_ratios=[4, 1], wspace=0.1, hspace=0.2)
            
            # 1. Original Image (Top Left)
            ax1 = plt.subplot(gs[0, 0])
            ax1.imshow(img)
            ax1.axis('off')
            ax1.set_title(f"Image: {image_name}")
            
            # Load depth map
            depth_filename = Path(image_name).stem + ".npy"
            depth_path = images_dir / "saved_depth_maps" / depth_filename
            depth_map = None
            
            # 2. Depth Map (Top Right)
            ax2 = plt.subplot(gs[0, 1])
            
            if depth_path.exists():
                try:
                    depth_map = np.load(depth_path)
                    
                    min_depth = np.min(depth_map)
                    max_depth = np.max(depth_map)
                    valid_depths = depth_map[np.isfinite(depth_map)]
                    avg_depth = np.mean(valid_depths) if valid_depths.size > 0 else float('nan')
                    
                    im = ax2.imshow(depth_map, cmap='turbo')
                    ax2.set_title(f"Depth Map (m)\nMin:{min_depth:.1f}, Max:{max_depth:.1f}, Avg:{avg_depth:.1f}")
                    ax2.axis('off')
                    
                    # Colorbar attached to the depth map axes
                    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, shrink=0.9)
                    cbar.set_label('Depth (m)', rotation=270, labelpad=15)
                    
                except Exception as e:
                    print(f"Error loading depth map {depth_path}: {e}")
                    ax2.text(0.5, 0.5, "Error loading depth map", ha='center', va='center')
                    ax2.axis('off')
            else:
                 ax2.text(0.5, 0.5, "Depth map not found", ha='center', va='center')
                 ax2.axis('off')
            
            # Prepare text
            if not display_results:
                 text_str = "No PlantNet Identifications found."
            else:
                text_str = f"All PlantNet Identifications ({len(display_results)}):\n"
                for i, (name, score, common_name) in enumerate(display_results):
                    prefix = ""
                    if crop_name and (crop_name.lower() in name.lower() or crop_name.lower() in common_name.lower()):
                        prefix = ">>> "
                    if common_name:
                        text_str += f"{prefix}{i+1}. {common_name} ({name}) - {score:.4f}\n"
                    else:
                        text_str += f"{prefix}{i+1}. {name} - {score:.4f}\n"
            
            # 3. Text Area (Bottom Left)
            ax3 = plt.subplot(gs[1, 0])
            ax3.axis('off')
            # Place text centered in this subplot
            ax3.text(0.5, 0.9, text_str, fontsize=12, ha='center', va='top', family='monospace')

            # 4. Histogram (Bottom Right)
            ax4 = plt.subplot(gs[1, 1])
            
            if depth_map is not None:
                # Filter for 0-30m
                valid_depths_sub30 = depth_map[(depth_map >= 0) & (depth_map <= 30)]
                if valid_depths_sub30.size > 0:
                     ax4.hist(valid_depths_sub30.flatten(), bins=30, range=(0, 30), color='gray', alpha=0.7)
                     ax4.set_title("Depth Distribution (0-30m)")
                     ax4.set_xlabel("Depth (m)")
                     ax4.grid(axis='y', alpha=0.3)
                else:
                     ax4.text(0.5, 0.5, "No points in 0-30m range", ha='center', va='center')
                     ax4.axis('off')
            else:
                ax4.axis('off')

        else:
            # Layout: 1 column
            plt.figure(figsize=(10, 8))
            ax1 = plt.subplot(1, 1, 1)
            ax1.imshow(img)
            ax1.axis('off')
            ax1.set_title(f"Image: {image_name}")
            
            if not display_results:
                 text_str = "No PlantNet Identifications found."
            else:
                text_str = f"All PlantNet Identifications ({len(display_results)}):\n"
                for i, (name, score, common_name) in enumerate(display_results):
                    prefix = ""
                    if crop_name and (crop_name.lower() in name.lower() or crop_name.lower() in common_name.lower()):
                        prefix = ">>> "
                    if common_name:
                        text_str += f"{prefix}{i+1}. {common_name} ({name}) - {score:.4f}\n"
                    else:
                        text_str += f"{prefix}{i+1}. {name} - {score:.4f}\n"

            # Use text attached to bottom of axis
            ax1.text(0.5, -0.05, text_str, fontsize=12, ha='center', va='top', transform=ax1.transAxes, family='monospace')
        
        # Get the max score from the displayed results for the filename
        max_score = 0.0
        if display_results:
            max_score = max(score for _, score, _ in display_results)

        # Create filename with score
        # Example: visu_12345_left_score_0.95.jpg
        stem = Path(image_name).stem
        suffix = Path(image_name).suffix
        outfile_name = f"visu_{stem}_score_{max_score:.2f}{suffix}"
        outfile = output_dir / outfile_name
        
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved visualization to {outfile}")

    print("Done!")
