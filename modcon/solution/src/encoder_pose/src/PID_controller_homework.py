#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np

# Lateral control

# TODO: write the PID controller using what you've learned in the previous activities

# Note: y_hat will be calculated based on your DeltaPhi() and poseEstimate() functions written previously 

def PIDController(
    v_0, # assume given (by the scenario)
    y_ref, # assume given (by the scenario)
    y_hat, # assume given (by the odometry)
    prev_e_y, # assume given (by the previous iteration of this function)
    prev_int_y, # assume given (by the previous iteration of this function)
    delta_t): # assume given (by the simulator)
    """
    Args:
        v_0 (:double:) linear Duckiebot speed.
        y_ref (:double:) reference lateral pose
        y_hat (:double:) the current estiamted pose along y.
        prev_e_y (:double:) tracking error at previous iteration.
        prev_int_y (:double:) previous integral error term.
        delta_t (:double:) time interval since last call.
    returns:
        v_0 (:double:) linear velocity of the Duckiebot 
        omega (:double:) angular velocity of the Duckiebot
        e_y (:double:) current tracking error (automatically becomes prev_e_y at next iteration).
        e_int_y (:double:) current integral error (automatically becomes prev_int_y at next iteration).
    """
    
    kp = 3.0
    ki = 0.0001
    kd = 25
    
    error = y_ref - y_hat
    e_int = prev_int_y + error * delta_t
    # anti-windup - preventing the integral error from growing too much
    e_int = max(min(e_int,2.0),-2.0)
    propTerm = error * kp
    intTerm = e_int * ki
    derivTerm = (error - prev_e_y) / delta_t * kd 
    
    
    
    # TODO: these are random values, you have to implement your own PID controller in here
    omega = propTerm + intTerm + derivTerm
    e_y = error
    e_int_y = e_int
    
    #print(f"\n\nomega : {omega} \npropTerm : {propTerm} \nintTerm : {intTerm} \nderivTerm : {derivTerm} \ny_hat: {y_hat} \ny_ref: {y_ref}\n")
    
    return [v_0, omega], e_y, e_int_y

