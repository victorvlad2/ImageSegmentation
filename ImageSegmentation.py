import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.io import imread, imshow
from skimage.transform import resize
import os
import pandas as pd
from skimage.transform import resize
from skimage.color import rgb2gray


image_dir = './flower_images'


def segment_image(image_path, n_segments=2):
    image = imread(image_path)
    image = resize(image, (128, 128), anti_aliasing=True)
    
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):  
        image = np.stack((image,) * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] > 3:  
        image = image[:, :, :3]

    image_array = image.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=n_segments, random_state=0).fit(image_array)
    labels = kmeans.predict(image_array)
    
    segmented_image = kmeans.cluster_centers_[labels].reshape((128, 128, 3))
    segmented_image = np.clip(segmented_image, 0, 1)
    
    return segmented_image

first_image_path = os.path.join(image_dir, os.listdir(image_dir)[0])
segmented_image = segment_image(first_image_path, n_segments=2)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
original_image = imread(first_image_path)
ax[0].imshow(original_image)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(segmented_image)
ax[1].set_title('Segmented Image')
ax[1].axis('off')

plt.show()
