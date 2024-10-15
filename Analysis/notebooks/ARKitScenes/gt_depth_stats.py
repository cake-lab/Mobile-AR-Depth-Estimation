import os
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count
import csv

# Function to collect depth statistics for a single sample
def collect_depth_values_from_sample(highres_depth_pth):
    # Load high-resolution depth map
    highres_depth = np.array(Image.open(highres_depth_pth)) / 1000  # Convert to meters
    valid_depth = highres_depth[highres_depth > 0]  # Exclude invalid depths (0 values)

    if valid_depth.size == 0:
        return None

    # Calculate statistics for this sample
    min_depth = np.min(valid_depth)
    max_depth = np.max(valid_depth)
    percentiles = np.percentile(valid_depth, [25, 50, 75, 90, 95, 99])

    return [highres_depth_pth, min_depth, max_depth] + list(percentiles)

# Function to collect all sample paths from the directories
def get_sample_paths(directories):
    sample_paths = []
    
    # Iterate through all directories and collect sample paths
    for i in directories:
        for scene in os.listdir(i):
            scene_path = os.path.join(i, scene, 'highres_depth')
            for sample in os.listdir(scene_path):
                highres_depth_pth = os.path.join(scene_path, sample)
                sample_paths.append(highres_depth_pth)
    
    return sample_paths

# Function to save depth statistics to CSV incrementally
def save_depth_statistics_to_csv(directories, output_csv):
    # Create the CSV file and write the header
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['sample_path', 'min_depth', 'max_depth', 'percentile_25', 'percentile_50', 'percentile_75', 'percentile_90', 'percentile_95', 'percentile_99'])

    # Get all sample paths
    sample_paths = get_sample_paths(directories)
    
    # Use multiprocessing to process samples and write results to CSV incrementally
    with Pool(processes=cpu_count()) as pool:
        with open(output_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Iterate over the results from the pool
            for result in pool.imap(collect_depth_values_from_sample, sample_paths):
                if result is not None:
                    writer.writerow(result)

upsampling_dir = '/mnt/IRONWOLF1/ashkan/data/ARKitScenes/upsampling'


training_dir_upsampling = os.path.join(upsampling_dir, 'Training')
validation_dir_upsampling = os.path.join(upsampling_dir, 'Validation')

directories = [training_dir_upsampling, validation_dir_upsampling]
output_csv = 'depth_statistics.csv'
save_depth_statistics_to_csv(directories, output_csv)

print(f"Depth statistics saved to {output_csv}")
