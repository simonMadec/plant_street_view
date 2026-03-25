"""
Mapillary (REST, gridded bbox) → CSV of thumb URLs → PlantNet on URLs → maps.

Requires config.yaml (site, bbox, Mapillary + PlantNet keys). Differs from the
removed fetch.py SDK path: this never downloads full panos to disk.
"""
import requests
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm as tqdm
import pandas as pd
import plotly.express as px
from pathlib import Path
import yaml
from utils.util import fetch_mapillary_images
from utils.util import parallel_process_images
from utils.util import create_map_visualization

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Mapillary API documentation : https://www.mapillary.com/developer/api-documentation?locale=fr_FR
# rate limits : https://www.mapillary.com/developer/api-documentation/?locale=fr_FR#rate-limits
MAPILLARY_TOKEN = config["fetch"]['mapillary']['api_key']

site = config['fetch']['site']

# 🔹 Generate sub-bounding boxes
min_lon = config['fetch']['bbox'][site]['min_lon']
min_lat = config['fetch']['bbox'][site]['min_lat']
max_lon = config['fetch']['bbox'][site]['max_lon']
max_lat = config['fetch']['bbox'][site]['max_lat']

STEP_SIZE = config['fetch']['step_size']

lon_grid = np.arange(min_lon, max_lon, STEP_SIZE)
lat_grid = np.arange(min_lat, max_lat, STEP_SIZE)

MAX_RETRIES = config['fetch']['mapillary']['MAX_RETRIES']
TIMEOUT = config['fetch']['mapillary']['TIMEOUT']
DELAY = config['fetch']['mapillary']['DELAY']  

PLANTNET_API_TOKEN = config['plantnet']['api_key']
chunk_size = config['plantnet']['chunk_size']

if Path(f"mapillary_data_{site}.csv").exists():
    results = pd.read_csv(f"mapillary_data_{site}.csv")
else:
    # 🔹 Initialize results DataFrame
    results = pd.DataFrame()

    # Use the function to fetch images
    results = fetch_mapillary_images(
        min_lon=min_lon, 
        min_lat=min_lat, 
        max_lon=max_lon, 
        max_lat=max_lat, 
        step_size=STEP_SIZE, 
        token=MAPILLARY_TOKEN,
        max_retries=MAX_RETRIES,
        timeout=TIMEOUT,
        delay=DELAY
    )
    
    if results.empty:
        print("No images found for this site")
    else:
        print(f"✅ Total images collected: {len(results)}")
        results.to_csv(f"mapillary_data_{site}.csv", index=False)

breakpoint()  # Remove for unattended runs; stops before PlantNet

results = pd.read_csv(f"mapillary_data_{site}.csv")
# ✅ Display summary
print(f"✅ Total images collected: {len(results)}")



# Create a scatter map visualization of the collected images
create_map_visualization(results, site)

results["best_match_scientific_name"] = None 
results["best_match_probability"] = None   
results["plantnet_data"] = None

# Create chunks and reset index for each to avoid KeyErrors
chunks = [results.iloc[i:i+chunk_size].reset_index(drop=True) for i in range(0, len(results), chunk_size)]

# Create an empty DataFrame to store results
final_results = pd.DataFrame()

# Process each chunk in parallel
step = 0

for chunk in tqdm(chunks):
    results_chunk = parallel_process_images(chunk, PLANTNET_API_TOKEN, n_jobs=8)
    final_results = pd.concat([final_results, results_chunk], ignore_index=True)
    # save final results
    final_results.to_csv("final_results_step_{}.csv".format(step), index=False)
    step += 1  # step index for checkpoint filenames

breakpoint()  # Remove for unattended runs
# save the results ! 
final_results.to_csv(f"final_results_{site}.csv", index=False)

results_notna = final_results[final_results["best_match_probability"].notna()]

probability_threshold = 0.0
results_filtered = results_notna[results_notna["best_match_probability"]>probability_threshold]

# Plotly expects columns named lat/lon; verify they match your CSV semantics
fig = px.scatter_mapbox(
    results_filtered,
    lat="lon",
    lon="lat",
    hover_name="best_match_scientific_name",
    hover_data=["best_match_probability"],
    color="best_match_scientific_name",
    zoom=11,
    mapbox_style="open-street-map"
)

fig.update_traces(marker=dict(size=15)) 

fig.update_layout(
    margin={"r":0,"t":0,"l":0,"b":0},  # Remove extra padding
    height=800)

fig.show()
