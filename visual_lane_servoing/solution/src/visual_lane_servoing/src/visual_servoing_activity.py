#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK

import cv2
import numpy as np


def get_steer_matrix_left_lane_markings(shape):
    """
        Args:
            shape: The shape of the steer matrix (tuple of ints)
        Return:
            steer_matrix_left_lane: The steering (angular rate) matrix for Braitenberg-like control 
                                    using the masked left lane markings (numpy.ndarray)
    """
    
    steer_matrix_left_lane = np.random.rand(*shape)

    return steer_matrix_left_lane

# In[ ]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK


def get_steer_matrix_right_lane_markings(shape):
    """
        Args:
            shape: The shape of the steer matrix (tuple of ints)
        Return:
            steer_matrix_right_lane: The steering (angular rate) matrix for Braitenberg-like control 
                                     using the masked right lane markings (numpy.ndarray)
    """
    
    steer_matrix_right_lane = np.random.rand(*shape)

    return steer_matrix_right_lane

# In[ ]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK

import cv2
import numpy as np

def detect_lane_markings(image):
    """
        Args:
            image: An image from the robot's camera in the BGR color space (numpy.ndarray)
        Return:
            left_masked_img:   Masked image for the dashed-yellow line (numpy.ndarray)
            right_masked_img:  Masked image for the solid-white line (numpy.ndarray)
    """

    # OpenCV uses BGR by default, whereas matplotlib uses RGB, so we generate an RGB version for the sake of visualization
    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to HSV for any color-based filtering
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Most of our operations will be performed on the grayscale version
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    print("image.shape: ", image.shape)
    print("imgrgb.shape: ", imgrgb.shape)
    print("imghsv.shape: ", imgrgb.shape)
    
    sigma = 6
    # Smooth the image using a Gaussian kernel
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)

    # Convolve the image with the Sobel operator (filter) to compute the numerical derivatives in the x and y directions
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)

    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

    # Compute the orientation of the gradients
    Gdir = cv2.phase(np.array(sobelx, np.float32), np.array(sobely, dtype=np.float32), angleInDegrees=True)

    
    threshold = 20
    mask_mag = (Gmag > threshold)

    white_lower_hsv = np.array([50, 8, 125])         # CHANGE ME
    white_upper_hsv = np.array([175, 45, 255])         # CHANGE ME
    yellow_lower_hsv = np.array([21, 145, 123])        # CHANGE ME
    yellow_upper_hsv = np.array([28, 255, 217])         # CHANGE ME

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    width = img.shape[1]
    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(width/2)):width + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(width/2))] = 0

    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)

    print("mask_ground.shape", mask_ground.shape)
    print("mask_mag.shape", mask_mag.shape)
    print("mask_sobelx_neg.shape", mask_sobelx_neg.shape)
    print("mask_sobely_neg.shape", mask_sobely_neg.shape)
    print("mask_yellow.shape", mask_yellow.shape)
    
    mask_left_edge = mask_ground * mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_ground * mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white

    #mask_left_edge = mask_ground  * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow * mask_left
    #mask_right_edge = mask_ground  * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white * mask_right
    
    h, w, _ = image.shape
    
    #mask_left_edge = np.random.rand(h, w)
    #mask_right_edge = np.random.rand(h, w)
    
    print("mask_left_edge.shape", mask_left_edge.shape)

    
    
    return (mask_left_edge, mask_right_edge)
