import numpy as np
import cv2

#lower_hsv = np.array([0, 100, 82])
#upper_hsv = np.array([10, 173, 255])

#lower_hsv = np.array([5, 92, 0])
#upper_hsv = np.array([91, 255, 255])

lower_hsv = np.array([0, 96, 0])
upper_hsv = np.array([35, 255, 255])

#maybe crank s min up to 127


def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """ Returns a 2D array """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    #     masked = cv2.bitwise_and(image, image, mask=mask)
    return mask
