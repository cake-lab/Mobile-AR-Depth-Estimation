import os
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import cv2

upsampling_dir = '/mnt/IRONWOLF1/ashkan/data/ARKitScenes/upsampling'

training_dir_upsampling = os.path.join(upsampling_dir, 'Training')
validation_dir_upsampling = os.path.join(upsampling_dir, 'Validation')


# Depth range definition based on dataset analysis
depth_ranges = [
    (0.1, 0.5),   # Very close range
    (0.5, 1.0),   # Close range
    (1.0, 1.5),   # Moderate range (includes 50th and 75th percentiles)
    (1.5, 2.0),   # Larger range (includes 90th to 99th percentiles)
    (2.0, 5.0),   # Farther range
    (5.0, 10.0),  # Very far range
    (10.0, 50.0)  # Extreme range
]

# Metric calculation functions
def calculate_rmse(gt, pred):
    """Root Mean Square Error (RMSE)"""
    return np.sqrt(np.mean((gt - pred) ** 2))

def calculate_abs_rel(gt, pred):
    """Absolute Relative Difference (AbsRel)"""
    return np.mean(np.abs(gt - pred) / gt)

def calculate_delta(gt, pred, threshold=1.25):
    """Delta accuracy with threshold"""
    ratio = np.maximum(gt / pred, pred / gt)
    return np.mean(ratio < threshold), np.mean(ratio < threshold**2), np.mean(ratio < threshold**3)

# Calculate depth range-specific accuracy
def calculate_depth_range_accuracy(highres_depth_valid, arkit_depth_valid, depth_ranges):
    """Calculate ARKit depth accuracy based on predefined depth ranges."""
    range_accuracies = {}
    
    for min_depth, max_depth in depth_ranges:
        # Create a mask for valid pixels within the depth range
        mask = (highres_depth_valid >= min_depth) & (highres_depth_valid < max_depth)
        
        if np.any(mask):
            # Calculate metrics only for the pixels within the current depth range
            range_rmse = calculate_rmse(highres_depth_valid[mask], arkit_depth_valid[mask])
            range_abs_rel = calculate_abs_rel(highres_depth_valid[mask], arkit_depth_valid[mask])
            delta1, delta2, delta3 = calculate_delta(highres_depth_valid[mask], arkit_depth_valid[mask])
        else:
            # If no valid pixels in the range, return NaN for the metrics
            range_rmse, range_abs_rel, delta1, delta2, delta3 = np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Store the metrics for the current range
        range_accuracies[f'rmse_{min_depth}-{max_depth}m'] = range_rmse
        range_accuracies[f'abs_rel_{min_depth}-{max_depth}m'] = range_abs_rel
        range_accuracies[f'delta1_{min_depth}-{max_depth}m'] = delta1
        range_accuracies[f'delta2_{min_depth}-{max_depth}m'] = delta2
        range_accuracies[f'delta3_{min_depth}-{max_depth}m'] = delta3

    return range_accuracies

# Sample processing function
def process_sample(highres_depth_pth, arkit_depth_pth):
    # Load high-resolution and ARKit depth maps
    highres_depth = np.array(Image.open(highres_depth_pth))
    arkit_depth = np.array(Image.open(arkit_depth_pth))

    # Downsample the highres depth to match ARKit resolution
    highres_depth_resized = cv2.resize(highres_depth, (arkit_depth.shape[1], arkit_depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Convert to meters
    highres_depth_resized = highres_depth_resized / 1000
    arkit_depth = arkit_depth / 1000
    
    # Mask out zero values (invalid depth points)
    valid_mask = (highres_depth_resized > 0) & (arkit_depth > 0)
    highres_depth_valid = highres_depth_resized[valid_mask]
    arkit_depth_valid = arkit_depth[valid_mask]
    
    if len(highres_depth_valid) == 0:
        return None
    
    # Calculate overall metrics
    rmse = calculate_rmse(highres_depth_valid, arkit_depth_valid)
    abs_rel = calculate_abs_rel(highres_depth_valid, arkit_depth_valid)
    delta1, delta2, delta3 = calculate_delta(highres_depth_valid, arkit_depth_valid)

    # Calculate depth range accuracy
    depth_range_metrics = calculate_depth_range_accuracy(highres_depth_valid, arkit_depth_valid, depth_ranges)

    # Collect all results for this sample
    results = {
        'highres_depth_path': highres_depth_pth,
        'arkit_depth_path': arkit_depth_pth,
        'rmse': rmse,
        'abs_rel': abs_rel,
        'delta1': delta1,
        'delta2': delta2,
        'delta3': delta3
    }

    # Add depth range metrics
    results.update(depth_range_metrics)
    
    return results

# Scene processing function (multi-threaded)
def process_scene(scene_path, executor):
    results = []
    for sample in os.listdir(scene_path):
        highres_depth_pth = os.path.join(scene_path, sample)
        arkit_depth_pth = highres_depth_pth.replace('highres_depth', 'lowres_depth')

        if os.path.exists(arkit_depth_pth):
            # Submit tasks to the executor
            future = executor.submit(process_sample, highres_depth_pth, arkit_depth_pth)
            results.append(future)
    
    return results

# Number of available threads/cores (adjust this dynamically)
num_threads = 32

# Initialize a list to store all results
results_list = []

# Multi-threaded processing
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    for i in [training_dir_upsampling, validation_dir_upsampling]:
        for scene in os.listdir(i):
            scene_path = os.path.join(i, scene, 'highres_depth')
            future_results = process_scene(scene_path, executor)
            
            for future in future_results:
                sample_results = future.result()
                if sample_results:
                    results_list.append(sample_results)

# Convert results to DataFrame for saving
df_results = pd.DataFrame(results_list)

# Save the results to a CSV file
df_results.to_csv('arkit_depth_accuracy_results.csv', index=False)

print("Results saved to 'arkit_depth_accuracy_results.csv'")
