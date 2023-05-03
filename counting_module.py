import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure

def identify_objects_above_threshold(img, threshold, min_size):

    # Threshold the image to create a binary image
    binary_img = np.zeros_like(img)
    binary_img[img > threshold] = 1

    # Label the connected components in the binary image
    label_img, num_labels = ndi.label(binary_img)

    # Find the properties of each connected component
    props = measure.regionprops(label_img, intensity_image=img)

    # Filter the connected components by tone value and size
    filtered_props = [prop for prop in props if prop.mean_intensity > threshold and prop.area > min_size]

    # Count the number of identified objects
    num_objects = len(filtered_props)

    # Determine the size of the identified objects
    object_sizes = [prop.area for prop in filtered_props]

    # Create a new image with the filtered objects
    filtered_img = np.zeros_like(img)
    for prop in filtered_props:
        filtered_img[label_img == prop.label] = 65535

    return num_objects, object_sizes, filtered_img