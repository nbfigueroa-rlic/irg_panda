#! /usr/bin/env python
from __future__ import print_function
import rospy, math, sys
from cartMotionControl.stateDep_class import CartesianMotionControl_StateDependent
from numpy import random as np_random

def rand_target_loc(np_random):
    '''
    generate random target location
    '''
    # x = np_random.uniform(low=0.05, high=0.5)
    # if np_random.randint(0, 2) == 0:
    #     x = -x
    # y = np_random.uniform(low=-0.3, high=0.2)

    # reference frames are different in gazebo and pybullet
    # y = np_random.uniform(low=0.05, high=0.5)
    y = np_random.uniform(low=0.10, high=0.5)
    if np_random.randint(0, 2) == 0:
        y = -y
    
    x = np_random.uniform(low = 0.3, high = 0.8)
    z = np_random.uniform(low=0.65,  high = 0.975)
    return x, y, z

# Global Variable
PI = math.pi

if __name__ == '__main__':

    rospy.init_node('cartesianVelocityControl_DS')    

    ####################################################
    #############    Parsing Parameters    #############
    ####################################################
    
    ####### DS variables #######
    DS_type    = rospy.get_param('~ds_type', 1)  # 1: Joint-space DS with joint-space target (JJ-DS)
                                                 # 2: Joint-space DS with task-space target  (JT-DS)
    
    #  Selected goal in joint space #
    goal       = rospy.get_param('~goal', 1)   

    # control for position + orientation
    # DS_attractor= [x, y, z, q_x, q_y, q_z, q_w]
    if goal == 1:
        ### Inside-Workspace (over-table) ###        
        DS_attractor   = [0.516, -0.000, 1.221, 0.983, -0.000, 0.183, 0.000] 

    elif goal == 2:    
        ### Out-of-Workspace (left-side) ###                
        DS_attractor   = [0.356, -0.605, 0.891, 0.999, 0.028, 0.040, -0.018]

    elif goal == 3: 
        DS_attractor   = [0.738, -0.197, 0.704, 0.683, 0.082, 0.679, -0.256]   
    
    elif goal == 4:
        DS_attractor   = [0.303, 0.547, 0.969, 0.999, -0.017, -0.031, -0.009]

    elif goal == 5:
        DS_attractor   = [0.627, 0.312, 0.703, 0.878, 0.044, 0.442, 0.181]

    # random positions from inside the box
    elif goal == 6:
        x,y,z = rand_target_loc(np_random)
        if y < 0: 
            DS_attractor   = [x, y, z, 0.683, 0.082, 0.679, -0.256]   
        else:
            DS_attractor   = [x, y, z, 0.878, 0.044, 0.442, 0.181]   

    # DS system matrix, gains for each joint error    
    A_p = [[5.0, 0, 0], 
           [0, 5.0, 0],
           [0, 0, 5.0]]

    # A_o = [[1.0, 0, 0], 
    #        [0, 1.0, 0],
    #        [0, 0, 1.0]]           

    A_o = [[3.0, 0, 0], 
           [0, 3.0, 0],
           [0, 0, 3.0]]           


    # Threshold for stopping DS     
    epsilon = 0.025
         
    ####### Motion Control Variables #######
    ctrl_rate   = 100 # 150hz
    ctrl_orient = 0
    
    ####### Initialize Class #######
    cartVelocityController = CartesianMotionControl_StateDependent(DS_type, A_p, A_o, DS_attractor, ctrl_rate, epsilon, ctrl_orient)

    ####### Run Control #######
    cartVelocityController.run()
