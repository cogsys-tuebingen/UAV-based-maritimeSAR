import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm

def convert_to_heatmap(image_path, output_path, colormap='jet'):
    """
    Convert a grayscale image to a heatmap image and save it.

    Parameters:
    - image_path: Path to the input grayscale image.
    - output_path: Path to save the output heatmap image.
    - colormap: Colormap to use for generating the heatmap.
    """
    # Load the image as a numpy array
    image = np.array(Image.open(image_path).convert('L'))
    
    # Create a figure and an axis (without displaying it)
    fig, ax = plt.subplots()
    cax = ax.imshow(image, cmap=colormap, interpolation='nearest')
    
    # Remove axis
    ax.axis('off')
    
    # Save the heatmap image to disk
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_directory(input_dir, output_dir, colormap='jet'):
    """
    Convert all grayscale images in a directory to heatmaps and save them.

    Parameters:
    - input_dir: Directory containing the grayscale images.
    - output_dir: Directory to save the heatmap images.
    - colormap: Colormap to use for generating the heatmaps.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate over all files in the input directory
    for filename in tqdm(os.listdir(input_dir)):
        # Construct full file path
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Convert the image to a heatmap and save it
        convert_to_heatmap(input_path, output_path, colormap=colormap)
        #print(f"Processed {filename}")

# Example usage
input_directory = '/tmp/heatmaps/' # Update this path
output_directory = '/tmp/coloured_heatmaps' # Update this path
process_directory(input_directory, output_directory, colormap='jet')

