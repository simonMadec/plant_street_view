"""
PlantNet on local projected JPGs (not URLs). Writes CSV columns plantnet_data, etc.

Uses file-based API; parallel + optional stats. Prefer config.yaml for API_KEY in new code.
"""
import requests
import os
import cv2
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Dict
import numpy as np
import ast
API_KEY = "2b10bfw41GZtUvKfJu5dRNFJu" #Jeremy's PlantNet API token

# --- Configuration ---
API_KEY = "2b10gGrBbgHiLFB0ZXCiAZTduO" #simon's s PlantNet API token

PROJECT = "all" # or # all
API_URL = f"https://my-api.plantnet.org/v2/identify/{PROJECT}"  # Use /all for worldwide flora


# --- Process all images in projected_views directory ---

projected_views_dir = Path("/home/simon/project/plant_street_view/images_kika/projected_views")
csv_path =  projected_views_dir. parent / "images_projected_metadata_kika_02122025.csv"


# --- Stats Tracking ---
class PlantNetStats:
    def __init__(self, stats_file="plantnet_stats.json"):
        self.stats_file = stats_file
        self.stats = self._load_stats()
    
    def _load_stats(self):
        """Load existing stats from file if it exists"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "total_calls": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "errors": 0,
            "rate_limit_remaining": None,
            "rate_limit_reset": None,
            "last_call_time": None,
            "last_response_headers": {}
        }
    
    def _save_stats(self):
        """Save stats to file"""
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def update(self, elapsed_time, response_headers=None, error=False, query_stats=None):
        """Update stats after an API call"""
        self.stats["total_calls"] += 1
        self.stats["total_time"] += elapsed_time
        self.stats["average_time"] = self.stats["total_time"] / self.stats["total_calls"]
        self.stats["last_call_time"] = datetime.now().isoformat()
        
        if error:
            self.stats["errors"] += 1
        
        if response_headers:
            # Track rate limit info from headers (common patterns)
            for header_name in ["X-RateLimit-Remaining", "RateLimit-Remaining", "X-Rate-Limit-Remaining"]:
                if header_name in response_headers:
                    try:
                        self.stats["rate_limit_remaining"] = int(response_headers[header_name])
                    except:
                        pass
            
            for header_name in ["X-RateLimit-Reset", "RateLimit-Reset", "X-Rate-Limit-Reset"]:
                if header_name in response_headers:
                    try:
                        self.stats["rate_limit_reset"] = int(response_headers[header_name])
                    except:
                        pass
            
            # Store all headers for inspection
            self.stats["last_response_headers"] = dict(response_headers)
        
        # Store query stats from API response if available
        if query_stats:
            if "last_query_stats" not in self.stats:
                self.stats["last_query_stats"] = {}
            self.stats["last_query_stats"] = query_stats
            # Track API processing time if available
            if "process" in query_stats:
                if "total_api_process_time" not in self.stats:
                    self.stats["total_api_process_time"] = 0.0
                self.stats["total_api_process_time"] += query_stats.get("process", 0)
                self.stats["average_api_process_time"] = self.stats["total_api_process_time"] / self.stats["total_calls"]
        
        self._save_stats()
    
    def print_stats(self):
        """Print current stats"""
        print("\n" + "="*50)
        print("📊 PlantNet API Usage Stats")
        print("="*50)
        print(f"Total API calls: {self.stats['total_calls']}")
        print(f"Total time (client): {self.stats['total_time']:.3f} seconds")
        print(f"Average time per call (client): {self.stats['average_time']:.3f} seconds")
        if self.stats.get("total_api_process_time"):
            print(f"Total API processing time: {self.stats['total_api_process_time']:.3f} seconds")
            print(f"Average API processing time: {self.stats.get('average_api_process_time', 0):.3f} seconds")
        print(f"Errors: {self.stats['errors']}")
        if self.stats.get("rate_limit_remaining") is not None:
            print(f"Rate limit remaining: {self.stats['rate_limit_remaining']}")
        if self.stats.get("rate_limit_reset") is not None:
            reset_time = datetime.fromtimestamp(self.stats['rate_limit_reset'])
            print(f"Rate limit resets at: {reset_time}")
        if self.stats.get("last_call_time"):
            print(f"Last call: {self.stats['last_call_time']}")
        if self.stats.get("last_query_stats"):
            last_stats = self.stats["last_query_stats"]
            print(f"\nLast API stats breakdown:")
            for key, value in last_stats.items():
                if isinstance(value, (int, float)) and key not in ['nb_images']:
                    print(f"  {key}: {value:.6f}s" if isinstance(value, float) else f"  {key}: {value}")
        print("="*50 + "\n")
    
    def get_stats(self):
        """Get stats dictionary"""
        return self.stats.copy()

# Initialize global stats tracker
stats_tracker = PlantNetStats()

# --- Helper Function to Extract Best Match ---
def extract_best_match(result_json):
    """Extract best match scientific name and probability from PlantNet response
    
    Response structure: {'results': [{'score': 0.14935, 'species': {...}, 'gbif': {...}}, ...]}
    """
    best_match_scientific_name = None
    best_match_probability = None
    
    if isinstance(result_json, dict):
        results_list = result_json.get('results', [])
        
        if results_list and isinstance(results_list, list):
            # Get the result with the highest score (first one is usually highest)
            best_result = max(results_list, key=lambda x: x.get('score', 0))
            best_match_probability = best_result.get('score', None)
            # Extract species info from the nested 'species' object
            species_info = best_result.get('species', {})
            if species_info:
                
                best_match_scientific_name = species_info.get('scientificNameWithoutAuthor', '')
    
    return best_match_scientific_name, best_match_probability

# --- Minimal Function ---
def identify_survey_image(api_key: str, url: str, image_file_path: str):
    
    params = {'api-key': api_key,
              'lang': 'fr',
              "nb-results": 20
              }
    
    # Parameters sent as form data (not URL parameters)
    data = {
        'organs': 'auto',
    }

    with open(image_file_path, 'rb') as image_file:
        # Let requests auto-detect the content type
        files = [
            ('images', (os.path.basename(image_file_path), image_file))
        ]
        
        start_time = time.time()
        error_occurred = False
        response_headers = None
        
        try:
            response = requests.post(url, params=params, files=files)
            response_headers = response.headers
            elapsed = time.time() - start_time
            response.raise_for_status() 
            result_json = response.json()
            
            # Extract query stats for tracking
            query_stats = None
            if isinstance(result_json, dict):
                query_stats = result_json.get('query', {}).get('stats', {})

            # Save JSON at the same place as the image
            json_file_path = os.path.splitext(image_file_path)[0] + ".json"
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=2)
            
            # Update stats with additional info from response
            stats_tracker.update(elapsed, response_headers, error_occurred, query_stats)
            
            return result_json
            
        except Exception as e:
            elapsed = time.time() - start_time
            error_occurred = True
            print(f"❌ Error: {e}")
            stats_tracker.update(elapsed, response_headers, error_occurred)
            return None
    
# Read CSV
print(f"📖 Reading CSV: {csv_path}")
df = pd.read_csv(csv_path)

# Initialize new columns if they don't exist
if 'plantnet_data' not in df.columns:
    df['plantnet_data'] = None
if 'best_match_scientific_name' not in df.columns:
    df['best_match_scientific_name'] = None
if 'best_match_probability' not in df.columns:
    df['best_match_probability'] = None

# df['plantnet_data'] = None
# df['best_match_probability'] = None
# df['best_match_scientific_name'] = None


if 'json_filename' in df.columns:
    df = df.drop(columns=['json_filename'])

# Set None for a random row in df["plantnet_data"]

# Get all image files in projected_views directory
image_files = list(projected_views_dir.glob("*.jpg"))
print(f"📸 Found {len(image_files)} images to process")

# Create a mapping from image ID to row index for faster lookup
print("🔍 Building image ID to CSV row mapping...")
id_to_row_idx = {}
for idx, row in df.iterrows():
    image_name = str(row.get('image_name', ''))
    if image_name and image_name.endswith('.jpg'):
        # Extract ID from image_name (e.g., "1235534165264921.jpg" -> "1235534165264921")
        img_id = image_name.replace('.jpg', '').split('_')[0]
        if img_id not in id_to_row_idx:
            id_to_row_idx[img_id] = []
        id_to_row_idx[img_id].append(idx)

# Count images that need processing vs already processed
print("\n📊 Analyzing processing status...")
already_processed = 0
need_processing = 0



for image_path in image_files:
    image_name = image_path.name
    row_idx = df[df['projected_file'] == image_name].index
    if len(row_idx) != 1:
        breakpoint()
        
    plantnet_data = df.at[row_idx[0], 'plantnet_data']
    
    # Try to convert string to dict if it's a string representation
    if isinstance(plantnet_data, str):
        try:
            dict_plantnet_data = ast.literal_eval(plantnet_data)
            df.at[row_idx[0], 'plantnet_data'] = dict_plantnet_data
            plantnet_data = dict_plantnet_data  # Update the variable for checking
        except (ValueError, SyntaxError):
            # If it's not a valid dict string, keep it as is
            pass

    # Check if already processed (has data starting with "skip" or any non-null data)
    # Handle NaN, None, empty strings, and the string "nan"
    if pd.isna(plantnet_data) or plantnet_data == "" or (isinstance(plantnet_data, str) and plantnet_data.strip().lower() == "nan"):
        # NaN, empty string, or "nan" string means needs processing
        need_processing += 1
    else:
        already_processed += 1

print(f"✅ Already processed: {already_processed} images")
print(f"🔄 Need processing: {need_processing} images")


for image_name_path in tqdm(image_files, desc="Processing images with PlantNet"):
        
    # Process ALL matching rows (both left and right if they exist)
    image_name = image_name_path.name
    row_idx = df[df['projected_file'] == image_name].index
    
    if len(row_idx) != 1:
        breakpoint()
    else:
        row_idx = row_idx[0]
        
    plantnet_data = df.at[row_idx, 'plantnet_data']
    
    # Skip if already processed (has plantnet_data that is not None or starts with "skip")
    if pd.notna(plantnet_data):
        if isinstance(plantnet_data, str) and plantnet_data.strip().lower().startswith('skip'):
            continue
        elif plantnet_data:  # Has any non-empty data
            continue
    
    # Only run identify_survey_image if plantnet_data is None or empty
    # Process image with PlantNet
    result_json = identify_survey_image(API_KEY, API_URL, str(image_name_path))
    
    if result_json is None:
        continue
    # Extract best match
    best_match_name, best_match_prob = extract_best_match(result_json)
    
    # Update CSV row
    df.at[row_idx, 'plantnet_data'] = json.dumps(result_json, ensure_ascii=False)
    df.at[row_idx, 'best_match_scientific_name'] = best_match_name
    df.at[row_idx, 'best_match_probability'] = best_match_prob
    
    # Save CSV periodically (every 10 images)
    if (image_files.index(image_name_path) + 1) % 10 == 0:
        df.to_csv(csv_path, index=False)
        print(f"💾 Progress saved to CSV")


# Final save
df.to_csv(csv_path, index=False)
print(f"✅ All results saved to {csv_path}")

# Print stats after processing
stats_tracker.print_stats()

