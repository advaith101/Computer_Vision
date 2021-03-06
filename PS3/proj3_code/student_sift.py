import numpy as np
import cv2
from proj3_code.student_harris import get_gradients
import math

def get_magnitudes_and_orientations(dx, dy):
    """
    This function will return the magnitudes and orientations of the
    gradients at each pixel location. 
    Args:
    -   dx: A numpy array of shape (m,n), representing x gradients in the image
    -   dy: A numpy array of shape (m,n), representing y gradients in the image

    Returns:
    -   magnitudes: A numpy array of shape (m,n), representing magnitudes of the
            gradients at each pixel location
    -   orientations: A numpy array of shape (m,n), representing angles of
            the gradients at each pixel location. angles should range from 
            -PI to PI.
 
    """
    magnitudes = []#placeholder
    orientations = []#placeholder

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    magnitudes = np.sqrt(dx**2 + dy**2)
    orientations = np.arctan2(dy, dx)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return magnitudes, orientations

def get_feat_vec(x,y,magnitudes, orientations,feature_width):
    """
    This function returns the feature vector for a specific interest point.
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described. The grid will extend
        feature_width/2 to the left of the "center", and feature_width/2 - 1 to the right
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions. The bin centers for the histogram 
        should be at -7pi/8,-5pi/8,...5pi/8,7pi/8. The histograms should be added
        to the feature vector left to right then row by row (reading order).  
    (3) Each feature should be normalized to unit length.
    (4) Each feature should be raised to a power less than one(use .9)

    For our tests, you do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though, so feel free to try it.
    The autograder will only check for each gradient contributing to a single bin.
    

    Args:
    -   x: a float, the x-coordinate of the interest point
    -   y: A float, the y-coordinate of the interest point
    -   magnitudes: A numpy array of shape (m,n), representing image gradients
            at each pixel location
    -   orientations: A numpy array of shape (m,n), representing gradient
            orientations at each pixel location
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.

    Returns:
    -   fv: A numpy array of shape (feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.

    A useful function to look at would be np.histogram.
    """

    fv = []#placeholder
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################

    #generate window of dimension feature_width*feature_width around (x,y)
    mag_window = magnitudes[y-feature_width//2:y+feature_width//2, x-feature_width//2:x+feature_width//2]
    ori_window = orientations[y-feature_width//2:y+feature_width//2, x-feature_width//2:x+feature_width//2]

    #divide window into 4*4 grids (each grid cell should be of dimension feature_width/4) and we are to assume feature width = 16
    r, h = mag_window.shape
    mag_split = split(mag_window, feature_width//4, feature_width//4)
    ori_split = split(ori_window, feature_width//4, feature_width//4)
    # mag_split = split(mag_window, 2, 2)
    # ori_split = split(ori_window, 2, 2)

    #do HoG - histogram of oriented gradients algorithm here for each 4x4 grid
    hists = np.empty((16,8))
    for i in range(0, mag_split.shape[0]):
        flat_mag = mag_split[i].flatten()
        flat_ori = ori_split[i].flatten()
        hist, bins = np.histogram(flat_ori, bins=8, range=(-math.pi,math.pi), weights=flat_mag)
        hists[i] = np.asarray(hist)

    #normalize
    flat_hists = hists.flatten()
    if np.linalg.norm(flat_hists) > 0:
        normalized = flat_hists / np.linalg.norm(flat_hists)
    else:
        normalized = flat_hists
    fv = normalized**0.9

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv


#HELPER FUNCTION - for splitting array into 4x4 blocks
def split(array, nrows, ncols):
    if array.size == 0:
        return array
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))



def get_features(image, x, y, feature_width):
    """
    This function returns the SIFT features computed at each of the input points
    You should code the above helper functions first, and use them below.
    You should also use your implementation of image gradients from before. 
    Hint: run get_feat_vec() with a loop
    Args:
    -   image: A numpy array of shape (m,n), the image
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.

    Returns:
    -   fvs: A numpy array of shape (k, feat_dim) representing all feature vectors.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #                                          #
    #############################################################################
    fvs = np.empty((x.shape[0], 128))
    ix, iy = get_gradients(image)
    magnitudes, orientations = get_magnitudes_and_orientations(ix, iy)
    for i in range(0, x.shape[0]):
        feat_vec = get_feat_vec(x[i], y[i], magnitudes, orientations, feature_width)
        fvs[i] = feat_vec
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fvs

