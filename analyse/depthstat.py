"""
Aggregate depth .npy stats and histograms; logs failed loads to results/stats/list_error.txt.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm



def save_list_error(list_error):
    with open("results/stats/list_error.txt", "w") as f:
        for item in list_error:
            f.write(f"{item}\n")
            
            
            
# Directory containing .npy files with depth maps
depths_dir = Path("/home/simon/project/plant_street_view/images_kika/projected_views/saved_depth_maps")

# List all .npy files in the directory
npy_files = list(depths_dir.glob("*.npy"))

# Efficiently accumulate all depth values (excluding inf, nan) from all files
all_depths = []
list_error = []
count_error = 0
for i,filename in tqdm(enumerate(npy_files), total=len(npy_files), desc="Loading depth maps"):
    try:
        arr = np.load(filename)
    except Exception as e:
        count_error += 1
        list_error.append(filename)
        print(f"Error loading {filename}: {e}")
        continue
    # Filter out nan and inf values
    valid = arr[np.isfinite(arr)]
    if valid.size > 0:
        all_depths.append(valid)

if len(all_depths) == 0:
    print("No valid depth data found.")
    exit(1)

save_list_error(list_error)


all_depths_flat = np.concatenate(all_depths)

# Compute statistics
min_depth = np.min(all_depths_flat)
max_depth = np.max(all_depths_flat)
mean_depth = np.mean(all_depths_flat)
median_depth = np.median(all_depths_flat)
std_depth = np.std(all_depths_flat)

print(f"Depth Statistics (meters):")
print(f"  Min:    {min_depth:.2f}")
print(f"  Max:    {max_depth:.2f}")
print(f"  Mean:   {mean_depth:.2f}")
print(f"  Median: {median_depth:.2f}")
print(f"  Std:    {std_depth:.2f}")
print(f"Number of errors: {count_error}")
# Plot histogram of depths
# Avoid memory issues by histogramming in chunks rather than concatenating all to one big array

# Define histogram params
bins = 100
depth_min = min_depth
depth_max = max_depth
hist = np.zeros(bins, dtype=np.int64)
bin_edges = np.linspace(depth_min, depth_max, bins + 1)

# Accumulate histogram per chunk
for chunk in all_depths:
    h, _ = np.histogram(chunk, bins=bin_edges)
    hist += h

# Convert bin centers for plotting
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

plt.figure(figsize=(10, 6))
plt.bar(bin_centers, hist, width=(bin_edges[1]-bin_edges[0]), color='blue', align='center', alpha=0.7)
plt.title("Histogram of Depth (meters) across all images")
plt.xlabel("Depth (meters)")
plt.ylabel("Pixel Count")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/stats/depth_histogram.png", dpi=300, bbox_inches="tight")
plt.show()

breakpoint()
print("Done")