import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.io import imread, imshow
from skimage.transform import resize
import os
import pandas as pd
from skimage.transform import resize
from skimage.color import rgb2gray


# Directory where images are stored
image_dir = './flower_images'


# Function to segment an image
def segment_image(image_path, n_segments=2):
    image = imread(image_path)
    # Ensure the image is resized correctly maintaining the 3 channels for RGB
    image = resize(image, (128, 128), anti_aliasing=True)
    
    # Check if the image is not RGB (i.e., grayscale or has an alpha channel) and convert it to RGB if necessary
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):  # Grayscale image
        # Convert grayscale to RGB by stacking the grayscale values along the depth
        image = np.stack((image,) * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] > 3:  # Image has more than 3 channels (e.g., RGBA)
        # Use only the first 3 channels (RGB)
        image = image[:, :, :3]

    # Flatten the image for KMeans
    image_array = image.reshape((-1, 3))
    
    # Apply KMeans to segment the image
    kmeans = KMeans(n_clusters=n_segments, random_state=0).fit(image_array)
    labels = kmeans.predict(image_array)
    
    # Create segmented image
    segmented_image = kmeans.cluster_centers_[labels].reshape((128, 128, 3))
    segmented_image = np.clip(segmented_image, 0, 1)  # Ensure the pixel values are valid
    
    return segmented_image

# Example: Segment the first image
first_image_path = os.path.join(image_dir, os.listdir(image_dir)[0])
segmented_image = segment_image(first_image_path, n_segments=2)

# Display the original and segmented image
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
original_image = imread(first_image_path)
ax[0].imshow(original_image)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(segmented_image)
ax[1].set_title('Segmented Image')
ax[1].axis('off')

plt.show()