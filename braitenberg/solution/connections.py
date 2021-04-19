from typing import Tuple

import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # we start with an array of 0s 
    res = np.zeros(shape=shape, dtype="float32")  

    # compute 1/2 of the image width, rounding down and forcing int type
    # Note: mid_width is 320 for Duckiebot but the code below works in the general case) 
    mid_width = int(np.floor(shape[1]/2))

    # write 1s in left side of image
    # This 'connects' the left motor with the left part of the image 
    res[:, 0:mid_width] = 1
    return res



def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    # we start with an array of 0s 
    res = np.zeros(shape=shape, dtype="float32")  

    # compute 1/2 of the image width, rounding down and forcing int type
    # Note: mid_width is 320 for Duckiebot but the code below works in the general case) 
    mid_width = int(np.floor(shape[1]/2))

    # write 1s in left side of image
    # This 'connects' the left motor with the left part of the image 
    res[:, mid_width:shape[1]] = 1
    return res
