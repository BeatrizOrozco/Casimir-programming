# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import skimage.filters as filters
import skimage.feature as feature
import skimage.morphology as morphology
import skimage.segmentation as segmentation
import skimage.measure as measure
import scipy.ndimage as ndi

#%% Function that calculates the curvatures defined by a set of points

def calculate_curvature(points):
    # Calculate first and second derivatives of the curve
    dx_dt, dy_dt = np.gradient(points, axis=0).T
    d2x_dt2, d2y_dt2 = np.gradient(np.array([dx_dt, dy_dt]), axis=0)

    # Calculate the curvature with the specified formula
    curvature = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**1.5
    
    return curvature

#%% Function that processes the image by smoothing, manipulating, and detecting edges
def preprocess_image(image, sigma=3, dilation_radius=2, min_size=200):
    # Smooth edges with Gaussian filter
    smooth_image = filters.gaussian(image, sigma=sigma)

    # Use Canny edge detection algorithm to detect edges 
    edges = feature.canny(smooth_image)

    # Dilate the edges using a disk-shaped element with the specified radius
    dilated_edges = morphology.dilation(edges, selem=morphology.disk(dilation_radius))

    # Fill the insides of the objects
    filled_edges = ndi.binary_fill_holes(dilated_edges)

    # Remove objects smaller than the specified minimum size
    cleaned_edges = morphology.remove_small_objects(filled_edges, min_size=min_size)
    return cleaned_edges

def segment_objects(cleaned_edges):
    '''
    Segments objects in a preprocessed binary image using the watershed algorithm.

    Input:
        cleaned_edges (NumPy array): A preprocessed binary image.

    Returns:
        NumPy array: A labeled image with segmented objects.
    '''
    # Compute the Euclidean distance transform of the cleaned_edges image
    distance = ndi.distance_transform_edt(cleaned_edges)

    # Find local maxima in the distance image, indicating the approximate centers of objects
    local_maxi = feature.peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=cleaned_edges)

    # Label the local maxima to create markers for the watershed algorithm
    markers = ndi.label(local_maxi)[0]

    # Apply the watershed algorithm using the markers and the inverted distance image
    labels = segmentation.watershed(-distance, markers, mask=cleaned_edges)
    return labels

def plot_segmented_objects(image, labels, contour_threshold=0.8):
    '''
    Plots the segmented objects along with their mean curvature on the original image.

    Input:
        image (NumPy array): The original grayscale image.
        labels (NumPy array): The labeled image with segmented objects.
        contour_threshold (float): Threshold for finding contours in the labeled image.
    '''
    # Find the contours of the segmented objects in the labeled image
    contours = measure.find_contours(labels, contour_threshold)

    # Calculate the curvature of each contour and its mean curvature
    curvatures = [calculate_curvature(contour) for contour in contours]
    mean_curvatures = [np.mean(curv) for curv in curvatures]

    # Create a plot with the original image and draw the contours
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')

    # Iterate through the contours and mean curvatures, plotting and annotating each object
    for contour, mean_curv in zip(contours, mean_curvatures):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        centroid = np.mean(contour, axis=0)
        ax.annotate(f"{mean_curv:.2f}", (centroid[1], centroid[0]), color='red', fontsize=12)

    # Add labels and titles to the plot
    ax.set_xlabel('X axis (pixels)')
    ax.set_ylabel('Y axis (pixels)')
    ax.set_title('Segmented objects with mean curvature')
    plt.show()
