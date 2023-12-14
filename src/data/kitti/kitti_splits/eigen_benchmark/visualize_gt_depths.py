import numpy as np
import os
import matplotlib.pyplot as plt

# Load the file
script_dir = os.path.dirname(os.path.abspath(__file__))
gt_depths_path = os.path.join(script_dir, 'gt_depths.npz')

data = np.load(gt_depths_path, allow_pickle=True)


# Print out the keys to see what's inside the .npz file
print("Arrays in the archive:", data.files)


depths = data['data']  # Replace 'depths' with the actual key.

# Check the shape of the array to understand its dimensions
print("Shape of the depths array:", depths.shape)

# Visualize the first depth map as an example (if it's a collection of depth maps)
plt.imshow(depths[0], cmap='inferno')  # You can play with different colormaps like 'viridis', 'plasma', 'gray', etc.
plt.colorbar()  # To show the color scale
plt.title("Depth Map Visualization")
plt.show()
