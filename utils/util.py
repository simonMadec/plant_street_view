"""
Shared helpers for main.py: Mapillary Graph API grid fetch, PlantNet-by-URL, map PNG.

PlantNet uses public image URLs (thumbnails), not local files—see plantnet_parrallel for files.
"""
import requests
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import urllib
from joblib import Parallel, delayed
import plotly.express as px

# Optionally display the figure (uncomment if running in a notebook)
def process_single_image(i, row, api_token):
    """
    Process a single image row from the DataFrame.
    Returns a dictionary with relevant results, keyed by index `i`.
    """
    url = row["image_url"]
    url = urllib.parse.quote(url, safe="")  # Encode the entire URL

    # Build the request
    request_url = (
        "https://my-api.plantnet.org/v2/identify/all" # or "https://my-api.plantnet.org/v2/identify/all" or "https://my-api.plantnet.org/v2/identify/k-caribbean"
        f"?images={url}"
        "&organs=auto"
        "&include-related-images=false"
        "&no-reject=false"
        "&nb-results=10"
        "&lang=fr"
        f"&api-key={api_token}"
    )

    response = requests.get(request_url)
    data = response.json()
    
    # Prepare output structure
    output = {
        "index": i,
        "plantnet_data": data,
        "best_match_scientific_name": None,
        "best_match_probability": None,
        "remaining_requests": data.get("remainingIdentificationRequests", "Unknown"),
    }

    # Extract the best match scientific name, if present
    if isinstance(data, dict) and "bestMatch" in data:
        output["best_match_scientific_name"] = data["bestMatch"]
        # Be sure 'results' is present and not empty
        if "results" in data and len(data["results"]) > 0:
            output["best_match_probability"] = data["results"][0].get("score", None)
        else:
            output["best_match_probability"] = None
    else:
        # No bestMatch found
        output["best_match_scientific_name"] = "Not Found"

    return output



def parallel_process_images(results, PLANTNET_API_TOKEN, n_jobs=4):
    """
    Parallelize the processing of images in the `results` DataFrame.
    Returns the updated DataFrame with new columns: 
    `plantnet_data`, `best_match_scientific_name`, and `best_match_probability`.
    """

    # Prepare a list of (index, row) to feed into joblib
    rows = [(i, results.loc[i]) for i in range(len(results))]

    # Parallel execution
    # Use 'backend="threading"' if you need I/O concurrency 
    # and the requests are primarily I/O-bound.
    outputs = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(process_single_image)(i, row, PLANTNET_API_TOKEN) 
        for (i, row) in tqdm(rows, desc="Submitting tasks")
    )
    
    # Collect the results into the DataFrame
    for out in outputs:
        i = out["index"]
        results.at[i, "plantnet_data"] = out["plantnet_data"]
        results.at[i, "best_match_scientific_name"] = out["best_match_scientific_name"]
        results.at[i, "best_match_probability"] = out["best_match_probability"]
        
        # If you still want to see how many requests remain (note:
        # it won't break early because the tasks are already launched in parallel)
        remaining_requests = out["remaining_requests"]
        if remaining_requests != "Unknown" and remaining_requests % 50 == 0:
            print(f"[Row {i}] Remaining requests: {remaining_requests}")

        # Hard to 'break' in parallel—once tasks are submitted, they run.
        # If you want to forcibly stop, you'd have to manage tasks in chunks or handle logic differently.

    return results



def fetch_mapillary_images(min_lon, min_lat, max_lon, max_lat, step_size, token, max_retries=3, timeout=30, delay=1):
    """
    Fetches images from Mapillary API within the specified bounding box.
    
    Args:
        min_lon (float): Minimum longitude of the bounding box
        min_lat (float): Minimum latitude of the bounding box
        max_lon (float): Maximum longitude of the bounding box
        max_lat (float): Maximum latitude of the bounding box
        step_size (float): Size of sub-bounding boxes
        token (str): Mapillary API token
        max_retries (int): Maximum number of retry attempts
        timeout (int): Request timeout in seconds
        delay (float): Delay between requests in seconds
        
    Returns:
        pd.DataFrame: DataFrame containing image data
    """
    results = pd.DataFrame()
    lon_grid = np.arange(min_lon, max_lon, step_size)
    lat_grid = np.arange(min_lat, max_lat, step_size)
    
    for lon in tqdm(lon_grid, desc="Processing longitude"):
        for lat in lat_grid:
            bbox = f"{lon},{lat},{lon+step_size},{lat+step_size}"
            url = f"https://graph.mapillary.com/images?access_token={token}&fields=id,geometry,thumb_1024_url,captured_at,is_pano,computed_compass_angle,computed_rotation,thumb_original_url,creator,thumb_2048_url&bbox={bbox}"#&is_pano=false"
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, timeout=timeout)
                    response.raise_for_status()
                    data = response.json()

                    if "data" in data and len(data["data"]) > 0:
                        temp_results = []

                        for image in data["data"]:
                            image_id = image["id"]
                            lat_img, lon_img = image["geometry"]["coordinates"]
                            capture_date = image.get("captured_at", "Unknown")
                            image_url = image.get("thumb_1024_url", "No image available")

                            temp_results.append({
                                "image_id": image_id,
                                "lat": lat_img,
                                "lon": lon_img,
                                "capture_date": capture_date,
                                "image_url": image_url
                            })

                        results = pd.concat([results, pd.DataFrame(temp_results)], ignore_index=True)
                        time.sleep(delay)
                    
                    print(f"🟢 {len(data['data'])} images found in bbox {bbox}")
                    break

                except requests.exceptions.RequestException as e:
                    print(f"🔴 Error on attempt {attempt + 1} for bbox {bbox}: {e}")
                    time.sleep(2)
    
    return results.reset_index(drop=True)


# Create a scatter map visualization of the collected images
def create_map_visualization(results, site):
    """
    Create a scatter map visualization of the collected images.
    
    Args:
        results (pd.DataFrame): DataFrame containing image data with lat and lon columns
        site (str): Name of the site for title and filename
        
    Returns:
        str: Path to the saved visualization file
    """
    
    # Create the scatter map
    # Note: lat/lon column names follow older code; confirm against DataFrame if map looks wrong
    fig = px.scatter_mapbox(
        results,
        lat="lon",
        lon="lat",
        zoom=12,
        mapbox_style="open-street-map"
    )

    # Adjust marker size for better visibility
    fig.update_traces(marker=dict(size=3))

    # Remove extra padding and set height
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        height=800,
        title=f"Mapillary Images - {site.capitalize()}"
    )

    # Save the visualization as a PNG file
    output_filename = f"mapillary_map_{site}.png"
    fig.write_image(output_filename)
    print(f"✅ Map visualization saved as {output_filename}")
    
    return output_filename