#!/usr/bin/env python
# coding: utf-8

# In[7]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK

#TODO: write a correct function

def DeltaPhi(encoder_msg, prev_ticks):
    """
        Args:
            encoder_msg: ROS encoder message (ENUM)
            prev_ticks: Previous tick count from the encoders (int)
        Return:
            rotation_wheel: Rotation of the wheel in radians (double)
            ticks: current number of ticks (int)
    """
    
    # TODO: these are random values, you have to implement your own solution in here
    ticks = encoder_msg.data    
    delta_phi = (ticks - prev_ticks)*2*np.pi/135

    return delta_phi, ticks

# In[8]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your odometry function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK

# TODO: write the odometry function

import numpy as np 

def poseEstimation( R, # radius of wheel (assumed identical) - this is fixed in simulation, and will be imported from your saved calibration for the physical robot
                    baseline_wheel2wheel, # distance from wheel to wheel; 2L of the theory
                    x_prev, # previous x estimate - assume given
                    y_prev, # previous y estimate - assume given
                    theta_prev, # previous orientation estimate - assume given
                    delta_phi_left, # left wheel rotation (rad)
                    delta_phi_right): # right wheel rotation (rad)
    
    """
        Calculate the current Duckiebot pose using the dead-reckoning approach.

        Returns x,y,theta current estimates:
            x_curr, y_curr, theta_curr (:double: values)
    """
    dl = R * delta_phi_left
    dr = R * delta_phi_right
    dA = (dr + dl) / 2
    dTheta = (dr - dl) / baseline_wheel2wheel
    
    # TODO: these are random values, you have to implement your own solution in here 
    theta_curr = theta_prev  +  dTheta
    x_curr = x_prev + dA * np.cos(theta_curr) 
    y_curr = y_prev + dA * np.sin(theta_curr)

    return x_curr, y_curr, theta_curr
