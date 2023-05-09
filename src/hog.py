import cv2
import numpy as np
from IPython.display import Image
from skimage.feature import local_binary_pattern
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import data, exposure


class HOG:

    hog = None

    def __init__(self):

        # Initialize SIFT object
        self.hog = cv2.HOG()
    
    def compute(gray_img):

        hog_features, hog_image = hog(gray_img, orientations=9 , pixels_per_cell= (8,8) ,
                              cells_per_block= (3,3) , 
                              visualize =True , block_norm = 'L2' , 
                               # transform_sqrt = False  
 
                              )
        # Apply 9ULBP
        # Define the parameters for LBP calculation
        radius = 1
        n_points = 8 * radius
        METHOD = 'uniform'

        # Calculate LBP image
        lbp_img = local_binary_pattern(gray_img, n_points, radius, METHOD)

        # Calculate 9ULBP histogram for each 8x8 cell
        cell_size = 8
        num_cells_x = lbp_img.shape[1] // cell_size
        num_cells_y = lbp_img.shape[0] // cell_size
        hist_size = 9
        histogram = np.zeros((num_cells_x * num_cells_y, hist_size), dtype=np.float32)
        for i in range(num_cells_y):
            for j in range(num_cells_x):
                cell = lbp_img[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                hist, _ = np.histogram(cell, bins=hist_size, range=(0, hist_size), density=True)
                histogram[i*num_cells_x + j, :] = hist

                # Normalize the histogram to L2 norm
                histogram /= np.sqrt(np.sum(np.power(histogram, 2)) + 1e-6)

                # Reshape the histogram as a feature vector
                ULBP_vector = histogram.ravel()

                # concatinate the two vector 
                feature_vector = np.concatenate((hog_features, ULBP_vector))
                return feature_vector




