"""
Threshold curves and species summaries from a projected PlantNet metadata CSV.

Writes figures under results/stats/; edit csv_path for your run.
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import ast

csv_path = "/data/data2/plant_street_view/images_kika/images_projected_metadata_kika.csv"

# Ensure results directory exists
results_dir = Path("results/stats")
results_dir.mkdir(parents=True, exist_ok=True)

# Read the data
df = pd.read_csv(csv_path)

# Print total number of images (rows)
print(f"Total images (rows in CSV): {len(df)}")

df['plantnet_max_score'] = df['best_match_probability']

# Compute proportions for thresholds from 0 to 1 step 0.05
thresholds = np.arange(0, 1.01, 0.05)
proportions = []
for t in thresholds:
    num_detected = (df['plantnet_max_score'] >= t).sum()
    prop = num_detected / len(df)
    proportions.append(prop)

# Plot
plt.figure(figsize=(8,5))
plt.plot(thresholds, proportions, marker='o', color='green', lw=2)
plt.xlabel('PlantNet detection threshold (score)', fontsize=13)
plt.ylabel('Proportion of images with detection ≥ threshold', fontsize=13)
plt.title('Proportion of PlantNet detections vs. detection threshold', fontsize=15)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0, 1.02)
plt.tight_layout()
threshold_fig_path = results_dir / "plantnet_detection_threshold.png"
plt.savefig(threshold_fig_path, dpi=300, bbox_inches="tight")
print(f"Detection-threshold figure saved to: {threshold_fig_path}")
plt.show()

# Create horizontal bar plot for species occurrence
# Check if the column exists
if 'best_match_scientific_name' in df.columns:
    # Filter out rows with missing species names
    df_species = df[df['best_match_scientific_name'].notna() & (df['best_match_scientific_name'] != '')].copy()
else:
    df_species = pd.DataFrame()  # Empty dataframe if column doesn't exist

if len(df_species) > 0:
    # Build dictionary mapping scientific names to common names
    def extract_common_name(plantnet_data_str, scientific_name):
        """Extract common name from PlantNet JSON data."""
        if pd.isna(plantnet_data_str) or not plantnet_data_str:
            return None
        
        try:
            # Try to parse as JSON string
            if isinstance(plantnet_data_str, str):
                # Try JSON first
                try:
                    data = json.loads(plantnet_data_str)
                except (json.JSONDecodeError, ValueError):
                    # Try ast.literal_eval as fallback
                    try:
                        data = ast.literal_eval(plantnet_data_str)
                    except (ValueError, SyntaxError):
                        return None
            else:
                data = plantnet_data_str
            
            # Extract common name from PlantNet response structure
            if isinstance(data, dict):
                results_list = data.get('results', [])
                if results_list and isinstance(results_list, list):
                    for result in results_list:
                        species_info = result.get('species', {})
                        if species_info:
                            sci_name = species_info.get('scientificNameWithoutAuthor', '')
                            if sci_name == scientific_name:
                                # Try different possible fields for common names
                                common_names = species_info.get('commonNames', [])
                                if not common_names:
                                    common_names = species_info.get('vernacularNames', [])
                                if not common_names:
                                    common_names = species_info.get('gbif', {}).get('vernacularNames', [])
                                
                                if common_names and isinstance(common_names, list) and len(common_names) > 0:
                                    # Return first common name (usually most common)
                                    return common_names[0]
        except Exception:
            pass
        
        return None
    
    # Build the mapping dictionary
    print("Building scientific name to common name mapping...")
    scientific_to_common = {}
    
    # Check if plantnet_data column exists
    if 'plantnet_data' in df_species.columns:
        for scientific_name in df_species['best_match_scientific_name'].unique():
            if pd.notna(scientific_name) and scientific_name:
                # Find first row with this scientific name that has plantnet_data
                matching_rows = df_species[
                    (df_species['best_match_scientific_name'] == scientific_name) &
                    (df_species['plantnet_data'].notna())
                ]
                
                if len(matching_rows) > 0:
                    plantnet_data = matching_rows.iloc[0]['plantnet_data']
                    common_name = extract_common_name(plantnet_data, scientific_name)
                    if common_name:
                        scientific_to_common[scientific_name] = common_name
    
    # Function to get display name (common name if available, otherwise scientific)
    def get_display_name(scientific_name):
        if scientific_name in scientific_to_common:
            return f"{scientific_to_common[scientific_name]} ({scientific_name})"
        return scientific_name
    
    # Aggregate stats per species: count and average probability
    species_stats = (
        df_species
        .groupby('best_match_scientific_name')['best_match_probability']
        .agg(['count', 'mean'])
        .rename(columns={'count': 'n_images', 'mean': 'avg_prob'})
        .sort_values('n_images', ascending=False)
    )

    # Show all species (not just top N)
    all_species = species_stats

    counts = all_species['n_images']
    avg_probs = all_species['avg_prob']

    # Map average probability to color (light to dark green)
    prob_min, prob_max = avg_probs.min(), avg_probs.max()
    denom = (prob_max - prob_min) if prob_max > prob_min else 1.0
    norm_probs = (avg_probs - prob_min) / denom
    colors = plt.cm.Greens(0.3 + 0.7 * norm_probs)  # avoid too pale colors

    # Create horizontal bar plot
    y_pos = range(len(all_species))
    plt.figure(figsize=(12, max(8, len(all_species) * 0.5)))
    plt.barh(y_pos, counts.values, color=colors, alpha=0.9)

    # Set y-axis labels to display names (common name + scientific name)
    display_names = [get_display_name(sci_name) for sci_name in all_species.index]
    plt.yticks(y_pos, display_names)

    # Add value labels on bars: count and avg probability
    for i, (cnt, p) in enumerate(zip(counts.values, avg_probs.values)):
        plt.text(
            cnt,
            i,
            f' {cnt} (avg p={p:.2f})',
            va='center',
            fontsize=9,
        )
    
    plt.xlabel('Number of Occurrences', fontsize=13)
    plt.ylabel('Species (Best Match)', fontsize=13)
    plt.title('All Species Occurrences (Based on Best Match)', fontsize=15)
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    species_fig_path = results_dir / "species_occurrence_barplot.png"
    plt.savefig(species_fig_path, dpi=300, bbox_inches="tight")
    print(f"Species-occurrence figure saved to: {species_fig_path}")
    plt.show()
    
    print(f"\nTotal species detected: {len(species_stats)}")
    print(f"Total images with species identification: {len(df_species)}")
    print(f"Species with common names found: {len(scientific_to_common)}")
else:
    print("\nNo species data available for plotting.")


# Create score distribution plots for specific species
# Extract all scores from PlantNet results (not just best match)
def extract_all_scores_for_species(df, species_keywords):
    """Extract all scores for species matching keywords from all PlantNet results."""
    scores_dict = {}
    
    if 'plantnet_data' not in df.columns:
        return scores_dict
    
    df_with_data = df[df['plantnet_data'].notna()].copy()
    
    for keyword in species_keywords:
        scores_dict[keyword] = []
        
        for idx, row in df_with_data.iterrows():
            plantnet_data_str = row['plantnet_data']
            
            if pd.isna(plantnet_data_str) or not plantnet_data_str:
                continue
            
            try:
                # Parse JSON data
                if isinstance(plantnet_data_str, str):
                    try:
                        data = json.loads(plantnet_data_str)
                    except (json.JSONDecodeError, ValueError):
                        try:
                            data = ast.literal_eval(plantnet_data_str)
                        except (ValueError, SyntaxError):
                            continue
                else:
                    data = plantnet_data_str
                
                # Extract all scores for this species
                if isinstance(data, dict):
                    results_list = data.get('results', [])
                    if results_list and isinstance(results_list, list):
                        for result in results_list:
                            species_info = result.get('species', {})
                            if species_info:
                                sci_name = species_info.get('scientificNameWithoutAuthor', '')
                                score = result.get('score', 0)
                                
                                # Check if species name contains keyword
                                if keyword.lower() in sci_name.lower():
                                    scores_dict[keyword].append(score)
            except Exception:
                pass
    
    return scores_dict

# Define species to plot
target_species = ['Sorghum', 'Gossypium', 'Tectona', 'Mangifera']

print("\nExtracting scores for target species...")
scores_by_species = extract_all_scores_for_species(df, target_species)

# Create distribution plots
for species_keyword in target_species:
    scores = scores_by_species.get(species_keyword, [])
    
    if len(scores) == 0:
        print(f"No scores found for {species_keyword}")
        continue
    
    scores = np.array(scores)
    
    # Create figure with line plot (density/histogram as line)
    plt.figure(figsize=(10, 6))
    # Calculate histogram with more bins for thinner bars
    n, bins = np.histogram(scores, bins=100, range=(0, 1))
    # Convert to percentage
    n_percent = (n / len(scores)) * 100
    # Use bin centers for x-axis
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # Plot as line
    plt.plot(bin_centers, n_percent, color='green', linewidth=2)
    # Add vertical red line at 0.02
    threshold = 0.02
    plt.axvline(x=threshold, color='red', linestyle='-', linewidth=2)
    
    # Calculate proportions before and after threshold
    prop_before = (scores < threshold).sum() / len(scores) * 100
    prop_after = (scores >= threshold).sum() / len(scores) * 100
    
    # Get max y value for text positioning
    y_max = np.max(n_percent)
    
    # Add text indicating threshold position
    plt.text(threshold, y_max * 0.95, f'Threshold: {threshold}', 
             rotation=90, verticalalignment='top', horizontalalignment='right',
             color='red', fontsize=11, fontweight='bold')
    
    # Add text with proportions
    text_str = f'Before threshold: {prop_before:.1f}%\nAfter threshold: {prop_after:.1f}%'
    plt.text(0.98, 0.95, text_str, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=11)
    
    plt.xlabel('PlantNet Score', fontsize=13)
    plt.ylabel('Frequency (%)', fontsize=13)
    plt.title(f'Score Distribution for {species_keyword}\n(n={len(scores)}, mean={np.mean(scores):.3f}, median={np.median(scores):.3f})', fontsize=15)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.xlim(0, 1)
    plt.ylim(0, None)
    plt.tight_layout()
    
    # Save figure
    species_safe = species_keyword.replace(' ', '_')
    dist_fig_path = results_dir / f"score_distribution_{species_safe}.png"
    plt.savefig(dist_fig_path, dpi=300, bbox_inches="tight")
    print(f"Score distribution figure saved to: {dist_fig_path}")
    plt.show()
    
    print(f"{species_keyword}: {len(scores)} scores, mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")

