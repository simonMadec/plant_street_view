"""
MoGe monocular depth per projected image; saves .npy under saved_depth_maps/.

Heavy: loads one HF model; GPU recommended. Used by visuresults when depth=True.
"""
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from moge.model.v2 import MoGeModel
from tqdm import tqdm

def compute_depth_image(image_path):
    """
    Loads an image, runs the MoGe model from Hugging Face,
    and returns a numpy array of estimated depth in meters for each pixel.
    
    Args:
        image_path (str or Path): Path to the input image.
    Returns:
        depth_map_meters (np.ndarray): 2D array (H, W) with depth per pixel, in meters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the MoGe model
    # The model is loaded from Hugging Face Hub: Ruicheng/moge-2-vitl-normal
    # Ensure you have installed moge: pip install git+https://github.com/microsoft/MoGe.git
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    model.eval()

    # Read the input image
    image_path_str = str(image_path)
    img_bgr = cv2.imread(image_path_str)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at {image_path_str}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor: (C, H, W), float32, [0, 1]
    input_tensor = torch.tensor(img_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    
    # Infer
    with torch.no_grad():
        output = model.infer(input_tensor)
        
    # Extract depth map (in meters)
    # The output dictionary contains 'depth' key with the depth map
    depth_map_meters = output['depth'].cpu().numpy()
    
    return depth_map_meters

def visualize_depth(image_path, depth_map, output_path):
    """
    Visualizes the depth map using matplotlib with a colorbar.
    Adds stats (Min, Max, Avg excluding Inf, Center) to the plot.
    
    Args:
        image_path (str or Path): Path to the original image.
        depth_map (np.ndarray): Depth map in meters.
        output_path (str or Path): Path to save the visualization.
    """
    # Read original image
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        tqdm.write(f"Could not read image {image_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Calculate stats
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    
    center_y, center_x = depth_map.shape[0]//2, depth_map.shape[1]//2
    center_depth = depth_map[center_y, center_x]
    
    # Filter out inf values for average calculation
    valid_depths = depth_map[np.isfinite(depth_map)]
    if valid_depths.size > 0:
        avg_depth = np.mean(valid_depths)
    else:
        avg_depth = float('nan')

    # Create figure
    plt.figure(figsize=(20, 10))

    # Plot Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')

    # Plot Depth Map
    # 'turbo' is a high-contrast colormap suitable for depth
    # It goes from Blue (close/low) to Red (far/high)
    plt.subplot(1, 2, 2)
    im = plt.imshow(depth_map, cmap='turbo') 
    plt.title(f"Depth Map (Meters)\nMin: {min_depth:.2f}m, Max: {max_depth:.2f}m, Avg: {avg_depth:.2f}m, Center: {center_depth:.2f}m")
    plt.axis('off')

    # Add Colorbar
    # shrink=0.7 makes it shorter
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04, shrink=0.7)
    cbar.set_label('Depth (m)', rotation=270, labelpad=15)
    
    # Save
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    
    # tqdm.write(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    # Input directory
    input_dir = Path("/data/data2/plant_street_view/images_kika/projected_views")
    
    
    # Output for numpy arrays (data)
    npy_output_dir = input_dir / "saved_depth_maps"
    npy_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output for visualizations
    vis_output_dir = Path("/home/simon/project/plant_street_view/results/depth")
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Numpy output directory: {npy_output_dir}")
    print(f"Visualization output directory: {vis_output_dir}")

    # Find all images
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(ext))
    
    # Sort to ensure consistent order
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print("No images found in input directory.")
    else:
        print(f"Found {len(image_files)} images. Starting processing...")

    for img_path in tqdm(image_files, desc="Processing Images"):
        # Check if output already exists
        npy_filename = img_path.stem + ".npy"
        npy_path = npy_output_dir / npy_filename
        
        if npy_path.exists():
            # tqdm.write(f"Skipping {img_path.name}, output already exists at {npy_path}")
            continue
            
        try:
            # Compute depth
            depth_map = compute_depth_image(img_path)
            
            # Save numpy array
            np.save(npy_path, depth_map)
            
            # Visualization
            vis_filename = f"vis_{img_path.name}"
            vis_path = vis_output_dir / vis_filename
            visualize_depth(img_path, depth_map, vis_path)
            
            # Optional: log success
            # tqdm.write(f"Saved {npy_path.name}")
            
        except Exception as e:
            tqdm.write(f"Error processing {img_path.name}: {e}")
            # import traceback
            # traceback.print_exc() # This might be too verbose for the progress bar
            tqdm.write("Continuing to next image...")

