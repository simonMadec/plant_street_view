"""Compare two projected_views directories (left/right pairs per Mapillary id)."""
import pandas as pd
from pathlib import Path
import re
from collections import defaultdict

# Paths
base_dir = Path("/data/data2/plant_street_view/images_kika/")
projected_views_dir1 = base_dir / "projected_views/"
projected_views_dir2 = base_dir / "photosphere" / "projected_views/"

print(f"{'='*70}")
print("COMPARING PROJECTED VIEWS DIRECTORIES")
print(f"{'='*70}\n")

print(f"Directory 1: {projected_views_dir1}")
print(f"Directory 2: {projected_views_dir2}\n")

# Scan both directories
def scan_directory(directory):
    """Scan directory and return dict: {image_id: {'left': path, 'right': path}}"""
    files = {}
    jpg_files = list(directory.glob("*.jpg"))
    
    for jpg_file in jpg_files:
        filename = jpg_file.name
        match = re.match(r'^(\d+)_(left|right)\.jpg$', filename)
        if match:
            image_id = match.group(1)
            view = match.group(2)
            
            if image_id not in files:
                files[image_id] = {}
            files[image_id][view] = str(jpg_file)
    
    return files

print("Scanning directories...")
files1 = scan_directory(projected_views_dir1)
files2 = scan_directory(projected_views_dir2)

print(f"\n{'='*70}")
print("STATISTICS")
print(f"{'='*70}")
print(f"Directory 1: {len(list((projected_views_dir1).glob('*.jpg')))} JPG files")
print(f"             {len(files1)} unique image IDs")
print(f"             {sum(1 for v in files1.values() if 'left' in v and 'right' in v)} complete pairs")
print(f"             {sum(1 for v in files1.values() if 'left' not in v or 'right' not in v)} incomplete pairs")

print(f"\nDirectory 2: {len(list((projected_views_dir2).glob('*.jpg')))} JPG files")
print(f"             {len(files2)} unique image IDs")
print(f"             {sum(1 for v in files2.values() if 'left' in v and 'right' in v)} complete pairs")
print(f"             {sum(1 for v in files2.values() if 'left' not in v or 'right' not in v)} incomplete pairs")

# Find differences
ids1 = set(files1.keys())
ids2 = set(files2.keys())

only_in_dir1 = ids1 - ids2
only_in_dir2 = ids2 - ids1
in_both = ids1 & ids2

print(f"\n{'='*70}")
print("OVERLAP ANALYSIS")
print(f"{'='*70}")
print(f"Images only in Directory 1: {len(only_in_dir1)}")
print(f"Images only in Directory 2: {len(only_in_dir2)}")
print(f"Images in both directories: {len(in_both)}")

# Check completeness in both
complete_in_both = 0
incomplete_in_both = []
incomplete_in_dir1 = []
incomplete_in_dir2 = []

for img_id in in_both:
    has_left1 = 'left' in files1[img_id]
    has_right1 = 'right' in files1[img_id]
    has_left2 = 'left' in files2[img_id]
    has_right2 = 'right' in files2[img_id]
    
    complete1 = has_left1 and has_right1
    complete2 = has_left2 and has_right2
    
    if complete1 and complete2:
        complete_in_both += 1
    else:
        if not complete1:
            incomplete_in_dir1.append(img_id)
        if not complete2:
            incomplete_in_dir2.append(img_id)
        incomplete_in_both.append((img_id, complete1, complete2))

print(f"\nComplete pairs in both directories: {complete_in_both}")
if incomplete_in_dir1:
    print(f"Incomplete in Directory 1: {len(incomplete_in_dir1)}")
if incomplete_in_dir2:
    print(f"Incomplete in Directory 2: {len(incomplete_in_dir2)}")

# Show samples
print(f"\n{'='*70}")
print("SAMPLE DIFFERENCES")
print(f"{'='*70}")

if only_in_dir1:
    print(f"\nFirst 10 image IDs only in Directory 1:")
    for img_id in list(only_in_dir1)[:10]:
        views = list(files1[img_id].keys())
        print(f"  - {img_id} (views: {', '.join(views)})")

if only_in_dir2:
    print(f"\nFirst 10 image IDs only in Directory 2:")
    for img_id in list(only_in_dir2)[:10]:
        views = list(files2[img_id].keys())
        print(f"  - {img_id} (views: {', '.join(views)})")

if incomplete_in_both[:5]:
    print(f"\nFirst 5 images incomplete in at least one directory:")
    for img_id, complete1, complete2 in incomplete_in_both[:5]:
        print(f"  - {img_id}: Dir1 complete={complete1}, Dir2 complete={complete2}")

# File size comparison for images in both
if in_both:
    print(f"\n{'='*70}")
    print("FILE SIZE COMPARISON (for images in both directories)")
    print(f"{'='*70}")
    
    size_differences = []
    for img_id in list(in_both)[:20]:  # Sample of 20
        if 'left' in files1[img_id] and 'left' in files2[img_id]:
            size1 = Path(files1[img_id]['left']).stat().st_size
            size2 = Path(files2[img_id]['left']).stat().st_size
            if size1 != size2:
                size_differences.append((img_id, 'left', size1, size2))
    
    if size_differences:
        print(f"\nFound {len(size_differences)} files with different sizes (showing first 5):")
        for img_id, view, size1, size2 in size_differences[:5]:
            diff = abs(size1 - size2)
            print(f"  - {img_id}_{view}: Dir1={size1:,} bytes, Dir2={size2:,} bytes (diff: {diff:,})")
    else:
        print("\nNo size differences found in sampled files.")

# Check for additional files/folders
print(f"\n{'='*70}")
print("ADDITIONAL CONTENT")
print(f"{'='*70}")

# Check for depth maps
depth_dir1 = projected_views_dir1 / "saved_depth_maps"
depth_dir2 = projected_views_dir2 / "saved_depth_maps"
if depth_dir1.exists():
    depth_files1 = list(depth_dir1.glob("*"))
    print(f"Directory 1 has 'saved_depth_maps/' folder with {len(depth_files1)} files")
    depth_types = {}
    for f in depth_files1:
        ext = f.suffix
        depth_types[ext] = depth_types.get(ext, 0) + 1
    print(f"  File types: {depth_types}")
else:
    print(f"Directory 1: No 'saved_depth_maps/' folder")
    
if depth_dir2.exists():
    depth_files2 = list(depth_dir2.glob("*"))
    print(f"Directory 2 has 'saved_depth_maps/' folder with {len(depth_files2)} files")
else:
    print(f"Directory 2: No 'saved_depth_maps/' folder")

# Check for JSON files
json_files1 = list(projected_views_dir1.glob("*.json"))
json_files2 = list(projected_views_dir2.glob("*.json"))
print(f"\nJSON files:")
print(f"  Directory 1: {len(json_files1)} JSON files")
print(f"  Directory 2: {len(json_files2)} JSON files")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Directory 1 has {len(ids1)} unique images")
print(f"Directory 2 has {len(ids2)} unique images")
print(f"They share {len(in_both)} common images")
print(f"Directory 2 has {len(only_in_dir2)} additional images not in Directory 1")
if only_in_dir1:
    print(f"Directory 1 has {len(only_in_dir1)} images not in Directory 2")

print(f"\nKey Differences:")
print(f"  - Directory 1: Has depth maps folder ({len(depth_files1) if depth_dir1.exists() else 0} files)")
print(f"  - Directory 1: Has {len(json_files1)} JSON files")
print(f"  - Directory 2: No depth maps, no JSON files")
print(f"  - Directory 2: Has {len(only_in_dir2)} more images than Directory 1")
print(f"{'='*70}\n")


