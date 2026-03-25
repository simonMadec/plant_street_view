"""
Rebuild projected-view metadata CSV by scanning photosphere/projected_views/*.jpg.

Fills rows from images_metadata_*.csv when ids match; use after reproject or to
repair CSVs without re-running projection.
"""
import pandas as pd
from pathlib import Path
import re

# Paths
source_metadata_csv = Path("/data/data2/plant_street_view/images_kika/images_metadata_kika2.csv")
base_dir = Path("/data/data2/plant_street_view/images_kika/")
projected_views_dir = base_dir / "photosphere" / "projected_views/"
output_csv = base_dir / "images_projected_metadata_kika2.csv"  # Output to main directory

print(f"Loading source metadata from: {source_metadata_csv}")
df_source = pd.read_csv(source_metadata_csv)
print(f"Source metadata has {len(df_source)} images")

# Create a dictionary for quick lookup by image ID
source_dict = {str(row['id']): row for _, row in df_source.iterrows()}
print(f"Created lookup dictionary with {len(source_dict)} entries")

# Scan ONLY photosphere/projected_views directory for all JPG files
print(f"\nScanning directory: {projected_views_dir}")
jpg_files = list(projected_views_dir.glob("*.jpg"))
print(f"Found {len(jpg_files)} JPG files")

# Extract image IDs and views from filenames
# Pattern: {image_id}_{view}.jpg where view is "left" or "right"
projected_data = []
image_ids_found = set()

for jpg_file in jpg_files:
    filename = jpg_file.name
    # Match pattern: {id}_left.jpg or {id}_right.jpg
    match = re.match(r'^(\d+)_(left|right)\.jpg$', filename)
    if match:
        image_id_str = match.group(1)
        view = match.group(2)
        image_ids_found.add(image_id_str)
        
        # Get metadata from source if available
        if image_id_str in source_dict:
            row_data = source_dict[image_id_str].copy()
        else:
            # Create minimal row if not in source metadata
            row_data = pd.Series({
                'id': int(image_id_str),
                'image_name': f"{image_id_str}.jpg"
            })
        
        # Add projected view information
        row_data["projected_view"] = view
        row_data["projected_file"] = filename
        row_data["projected_path"] = str(jpg_file.absolute())
        
        projected_data.append(row_data)
    else:
        print(f"Warning: File doesn't match expected pattern: {filename}")

print(f"\nExtracted {len(projected_data)} projected image entries")
print(f"Unique image IDs found: {len(image_ids_found)}")

# Convert to DataFrame
if projected_data:
    # Ensure all rows have the same columns
    all_columns = set()
    for row in projected_data:
        all_columns.update(row.index)
    
    # Create DataFrame with all columns
    projected_df = pd.DataFrame(projected_data)
    
    # Sort by image ID and view (left before right)
    projected_df = projected_df.sort_values(['id', 'projected_view'])
    
    # Save to CSV
    projected_df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved CSV to: {output_csv}")
    print(f"   Total rows: {len(projected_df)}")
    
    # Show statistics
    if 'projected_view' in projected_df.columns:
        print(f"\nView distribution:")
        print(projected_df['projected_view'].value_counts())
    
    # Check for missing metadata
    missing_metadata = [img_id for img_id in image_ids_found if img_id not in source_dict]
    if missing_metadata:
        print(f"\n⚠️  Warning: {len(missing_metadata)} image IDs not found in source metadata:")
        print(f"   First 10: {missing_metadata[:10]}")
    else:
        print(f"\n✅ All image IDs found in source metadata")
    
    print(f"\n{'='*60}")
    print(f"CSV created successfully!")
    print(f"{'='*60}")
else:
    print("\n❌ No projected images found!")


