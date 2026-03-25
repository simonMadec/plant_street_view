"""Sanity-check a projected metadata CSV (row counts, column presence, file paths)."""
import pandas as pd
from pathlib import Path

# Path to the CSV file - checking the non-photosphere directory
csv_path = Path("/data/data2/plant_street_view/images_kika/projected_views/images_projected_metadata_kika2.csv")

print(f"Loading CSV: {csv_path}")
print(f"File exists: {csv_path.exists()}")

if not csv_path.exists():
    print(f"ERROR: File not found at {csv_path}")
    exit(1)

# Load the CSV
df = pd.read_csv(csv_path)

print(f"\n{'='*60}")
print("CSV FILE STATISTICS")
print(f"{'='*60}")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nColumns: {list(df.columns)}")

# Check for projected_view column if it exists
if 'projected_view' in df.columns:
    print(f"\nProjected view distribution:")
    print(df['projected_view'].value_counts())
    
    # Count unique image IDs
    if 'id' in df.columns:
        unique_ids = df['id'].nunique()
        print(f"\nUnique image IDs: {unique_ids}")
        print(f"Expected: {len(df) / 2:.0f} (if 2 views per image)")
elif 'id' in df.columns:
    unique_ids = df['id'].nunique()
    print(f"\nUnique image IDs: {unique_ids}")

# Check for file paths
if 'projected_path' in df.columns or 'projected_file' in df.columns:
    print(f"\n{'='*60}")
    print("FILE VERIFICATION")
    print(f"{'='*60}")
    
    if 'projected_path' in df.columns:
        file_exists = df['projected_path'].apply(lambda x: Path(x).exists() if pd.notna(x) else False)
        existing_files = file_exists.sum()
        print(f"Files that exist on disk: {existing_files} / {len(df)}")
        print(f"Missing files: {len(df) - existing_files}")
        
        missing = df[~file_exists]
        if len(missing) > 0:
            print(f"\nMissing file breakdown:")
            if 'projected_view' in missing.columns:
                print(missing['projected_view'].value_counts())
            
            if 'id' in missing.columns:
                missing_ids = missing['id'].unique()
                print(f"\nUnique image IDs with missing files: {len(missing_ids)}")
                print(f"First 10 missing image IDs:")
                for img_id in missing_ids[:10]:
                    missing_views = missing[missing['id'] == img_id]['projected_view'].tolist()
                    print(f"  - {img_id}: missing {', '.join(missing_views)}")
            
            print(f"\nFirst 5 missing file paths:")
            print(missing[['id', 'projected_view', 'projected_path']].head() if 'id' in missing.columns else missing[['projected_path']].head())

# Show first few rows
print(f"\n{'='*60}")
print("FIRST 5 ROWS")
print(f"{'='*60}")
print(df.head())

# Show data types
print(f"\n{'='*60}")
print("DATA TYPES")
print(f"{'='*60}")
print(df.dtypes)

print(f"\n{'='*60}")
