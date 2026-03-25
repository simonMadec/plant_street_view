"""Diff two projected-metadata CSVs (columns, row counts, key columns)."""
import pandas as pd
from pathlib import Path

# Paths to the two CSV files
csv1_path = Path("/data/data2/plant_street_view/images_kika/projected_views/images_projected_metadata_kika2.csv")
# Check if the second CSV exists, if not, try to find it
csv2_path = Path("/data/data2/plant_street_view/images_kika/images_projected_metadata_kika2.csv")
if not csv2_path.exists():
    # Try photosphere location
    csv2_path = Path("/data/data2/plant_street_view/images_kika/photosphere/projected_views/images_projected_metadata_kika2.csv")

print(f"{'='*70}")
print("COMPARING CSV FILES")
print(f"{'='*70}\n")

print(f"CSV 1: {csv1_path}")
print(f"CSV 2: {csv2_path}\n")

# Check if files exist
if not csv1_path.exists():
    print(f"ERROR: CSV 1 not found at {csv1_path}")
    exit(1)

if not csv2_path.exists():
    print(f"ERROR: CSV 2 not found at {csv2_path}")
    exit(1)

# Load both CSVs
print("Loading CSV files...")
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

print(f"\n{'='*70}")
print("BASIC STATISTICS")
print(f"{'='*70}")
print(f"CSV 1 rows: {len(df1)}")
print(f"CSV 2 rows: {len(df2)}")
print(f"Difference: {abs(len(df1) - len(df2))} rows")

print(f"\nCSV 1 columns: {len(df1.columns)}")
print(f"CSV 2 columns: {len(df2.columns)}")

print(f"\nCSV 1 columns: {list(df1.columns)}")
print(f"\nCSV 2 columns: {list(df2.columns)}")

# Check if they have the same columns
if set(df1.columns) == set(df2.columns):
    print("\n✅ Both CSVs have the same columns")
else:
    only_in_csv1 = set(df1.columns) - set(df2.columns)
    only_in_csv2 = set(df2.columns) - set(df1.columns)
    if only_in_csv1:
        print(f"\n⚠️  Columns only in CSV 1: {only_in_csv1}")
    if only_in_csv2:
        print(f"\n⚠️  Columns only in CSV 2: {only_in_csv2}")

# Check unique image IDs if 'id' column exists
if 'id' in df1.columns and 'id' in df2.columns:
    print(f"\n{'='*70}")
    print("IMAGE ID COMPARISON")
    print(f"{'='*70}")
    ids1 = set(df1['id'].astype(str))
    ids2 = set(df2['id'].astype(str))
    
    print(f"Unique image IDs in CSV 1: {len(ids1)}")
    print(f"Unique image IDs in CSV 2: {len(ids2)}")
    
    only_in_csv1 = ids1 - ids2
    only_in_csv2 = ids2 - ids1
    in_both = ids1 & ids2
    
    print(f"\nImage IDs only in CSV 1: {len(only_in_csv1)}")
    print(f"Image IDs only in CSV 2: {len(only_in_csv2)}")
    print(f"Image IDs in both: {len(in_both)}")
    
    if only_in_csv1:
        print(f"\nFirst 10 image IDs only in CSV 1:")
        for img_id in list(only_in_csv1)[:10]:
            print(f"  - {img_id}")
    
    if only_in_csv2:
        print(f"\nFirst 10 image IDs only in CSV 2:")
        for img_id in list(only_in_csv2)[:10]:
            print(f"  - {img_id}")

# Check projected_view distribution
if 'projected_view' in df1.columns and 'projected_view' in df2.columns:
    print(f"\n{'='*70}")
    print("PROJECTED VIEW DISTRIBUTION")
    print(f"{'='*70}")
    print("CSV 1:")
    print(df1['projected_view'].value_counts())
    print("\nCSV 2:")
    print(df2['projected_view'].value_counts())

# Check projected_path if it exists
if 'projected_path' in df1.columns and 'projected_path' in df2.columns:
    print(f"\n{'='*70}")
    print("PATH ANALYSIS")
    print(f"{'='*70}")
    
    # Extract directory from paths
    def get_dir(path):
        return str(Path(path).parent)
    
    dirs1 = df1['projected_path'].apply(get_dir).unique()
    dirs2 = df2['projected_path'].apply(get_dir).unique()
    
    print(f"Directories in CSV 1 paths:")
    for d in dirs1:
        count = (df1['projected_path'].apply(get_dir) == d).sum()
        print(f"  - {d} ({count} files)")
    
    print(f"\nDirectories in CSV 2 paths:")
    for d in dirs2:
        count = (df2['projected_path'].apply(get_dir) == d).sum()
        print(f"  - {d} ({count} files)")

# Compare file existence
if 'projected_path' in df1.columns and 'projected_path' in df2.columns:
    print(f"\n{'='*70}")
    print("FILE EXISTENCE CHECK")
    print(f"{'='*70}")
    
    files_exist1 = df1['projected_path'].apply(lambda x: Path(x).exists()).sum()
    files_exist2 = df2['projected_path'].apply(lambda x: Path(x).exists()).sum()
    
    print(f"CSV 1: {files_exist1} / {len(df1)} files exist on disk")
    print(f"CSV 2: {files_exist2} / {len(df2)} files exist on disk")

# Sample rows comparison
print(f"\n{'='*70}")
print("SAMPLE ROWS")
print(f"{'='*70}")
print("\nCSV 1 - First 3 rows:")
print(df1.head(3)[['id', 'projected_view', 'projected_path']] if 'id' in df1.columns else df1.head(3))

print("\nCSV 2 - First 3 rows:")
print(df2.head(3)[['id', 'projected_view', 'projected_path']] if 'id' in df2.columns else df2.head(3))

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"CSV 1 location: {csv1_path.parent}")
print(f"CSV 1 size: {len(df1)} rows, {len(df1.columns)} columns")
if 'id' in df1.columns:
    print(f"CSV 1 unique images: {df1['id'].nunique()}")

print(f"\nCSV 2 location: {csv2_path.parent}")
print(f"CSV 2 size: {len(df2)} rows, {len(df2.columns)} columns")
if 'id' in df2.columns:
    print(f"CSV 2 unique images: {df2['id'].nunique()}")

if len(df1) == len(df2):
    print("\n✅ Both CSVs have the same number of rows")
else:
    print(f"\n⚠️  Different number of rows: {abs(len(df1) - len(df2))} difference")

if 'id' in df1.columns and 'id' in df2.columns:
    if ids1 == ids2:
        print("✅ Both CSVs contain the same image IDs")
    else:
        print(f"⚠️  Different image IDs: {len(only_in_csv1)} only in CSV 1, {len(only_in_csv2)} only in CSV 2")

print(f"{'='*70}\n")


