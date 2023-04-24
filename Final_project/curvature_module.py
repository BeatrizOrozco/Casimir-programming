# -*- coding: utf-8 -*-
"""
Spyder Editor

Module containing functions to compute the mean curvature of objects in a .tif image file.

The code was inspired by:
    https://ojskrede.github.io/inf4300/exercises/week_11/
    https://github.com/jmschabdach/caulobacter-curvature/blob/master/Identifying%20and%20Measuring%20the%20Curvature%20of%20Caulobacter%20Cells.ipynb
    https://stackoverflow.com/questions/9137216/python-edge-detection-and-curvature-calculation
    https://scikit-image.org/skimage-tutorials/lectures/1_image_filters.html
    https://towardsdatascience.com/image-segmentation-using-pythons-scikit-image-module-533a61ecc980

"""
import numpy as np
import matplotlib.pyplot as plt
import skimage.filters as filters
import skimage.feature as feature
import skimage.morphology as morphology
import skimage.segmentation as segmentation
import skimage.measure as measure
import scipy.ndimage as ndi

#%% Function to calculates the curvatures defined by a set of points

def calculate_curvature(points):
    # Calculate first and second derivatives of the curve
    dx_dt, dy_dt = np.gradient(points, axis=0).T
    d2x_dt2, d2y_dt2 = np.gradient(np.array([dx_dt, dy_dt]), axis=0)

    # Calculate the curvature with the specified formula
    curvature = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**1.5
    
    return curvature

#%% Function to pre-processes image; e.g., smoothing, manipulating, and detecting edges

def preprocess_image(image, sigma, dilation_radius, min_size):
    # Smooth edges with Gaussian filter
    smooth_image = filters.gaussian(image, sigma=sigma)

    # Canny edge detection algorithm to detect edges 
    edges = feature.canny(smooth_image)

    # Dilate edges using disk-shaped element with specified radius
    dilated_edges = morphology.dilation(edges, selem=morphology.disk(dilation_radius))

    # Fill the insides of the objects
    filled_edges = ndi.binary_fill_holes(dilated_edges)

    # Remove objects smaller than the specified minimum size
    cleaned_edges = morphology.remove_small_objects(filled_edges, min_size=min_size)
    
    return cleaned_edges

#%% Function to segment pre-processed image using the Watershed algorithm 

def segment_objects(cleaned_edges):
    # Euclidean distance transform of image
    distance = ndi.distance_transform_edt(cleaned_edges)

    # Local maxima (of distance image), approximates the centers of objects
    local_max = feature.peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=cleaned_edges)

    # Create markers for the watershed algorithm from the local maxima
    markers = ndi.label(local_max)[0]

    # Apply the watershed algorithm using the markers and the inverted distance image
    labels = segmentation.watershed(-distance, markers, mask=cleaned_edges)
    
    return labels

#%% Function to plot segmented objects along with their mean curvature on the original image. 

def plot_segmented_objects(image, labels, contour_threshold):
    # Find the contours of the segmented objects
    contours = measure.find_contours(labels, contour_threshold)

    # Calculate the each contour's curvature and their mean curvature
    curvatures = [calculate_curvature(contour) for contour in contours]
    mean_curvatures = [np.mean(curv) for curv in curvatures]

    # Plot with the original image and draw the contours
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')

    # Go through each object and plot with annotations
    for contour, mean_curv in zip(contours, mean_curvatures):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        center = np.mean(contour, axis=0)
        ax.annotate(f"{mean_curv:.2f}", (center[1], center[0]), color='red', fontsize=12)

    # Labels and titles to the plot
    ax.set_title('Segmented objects with mean curvature')
    ax.set_xlabel('Number of pixels')
    ax.set_ylabel('Number of pixels')
    plt.show()
