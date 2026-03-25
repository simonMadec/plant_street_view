"""
Equirectangular panos (photosphere/*.jpg) → left/right perspective JPGs + CSV.

Expects images_metadata_*.csv with Mapillary ids; writes two rows per pano
(left/right). heading_deg is fixed at 270° here—swap for row['compass'] if needed.
"""
import cv2
import numpy as np
import py360convert
from utils.jeremy360 import extract_left_right_views
import time
from pathlib import Path
import pandas as pd

# Paths
image_dir = Path("/data/data2/plant_street_view/images_kika/photosphere/")
metadata_csv = image_dir.parent / "images_metadata_kika2.csv"
output_dir = image_dir / "projected_views"
out_metadata_csv = output_dir / "images_projected_metadata_kika2.csv"


# Create output directory if it doesn't exist (don't remove existing files)
output_dir.mkdir(parents=True, exist_ok=True)

# Read metadata
df = pd.read_csv(metadata_csv)

projected_metadata = []

for idx, row in df.iterrows():
    image_id = row['id']
    image_filename = f"{image_id}.jpg"
    image_path = image_dir / image_filename

    if not image_path.is_file():
        print(f"File missing: {image_path}")
        continue

    try:
        heading_deg = 270.0  # TODO: use row compass (e.g. computed_compass_angle) when reliable
    except Exception:
        print(f"Could not parse compass_angle for {image_id}")
        continue
    
    # Setup output file names
    left_img_fn = f"{image_id}_left.jpg"
    right_img_fn = f"{image_id}_right.jpg"
    left_img_path = output_dir / left_img_fn
    right_img_path = output_dir / right_img_fn

    # Check if images are already projected
    if left_img_path.exists() and right_img_path.exists():
        print(f"✅ {image_id}: already processed, skipping projection")
    else:
        # Project images if they don't exist
        start_time = time.time()
        try:
            left_img, right_img = extract_left_right_views(
                pano_path=str(image_path),
                heading_deg=270.0,
                fov_h_deg=120.0,
                fov_v_deg=90.0,
                pitch_deg=7.0,
                out_width=1280,
                out_height=768,
            )
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            breakpoint()
            continue
        elapsed = time.time() - start_time
        print(f"⏱️ {image_id}: extract_left_right_views took {elapsed:.3f} seconds")

        cv2.imwrite(str(left_img_path), left_img)
        cv2.imwrite(str(right_img_path), right_img)

    # Add to new metadata
    row_left = row.copy()
    row_left["projected_view"] = "left"
    row_left["projected_file"] = left_img_fn
    row_left["projected_path"] = str(left_img_path)

    row_right = row.copy()
    row_right["projected_view"] = "right"
    row_right["projected_file"] = right_img_fn
    row_right["projected_path"] = str(right_img_path)

    projected_metadata.extend([row_left, row_right])

# Write new metadata CSV
meta_out_df = pd.DataFrame(projected_metadata)
meta_out_df.to_csv(out_metadata_csv, index=False)

print(f"Exported projected image metadata to {out_metadata_csv}")
