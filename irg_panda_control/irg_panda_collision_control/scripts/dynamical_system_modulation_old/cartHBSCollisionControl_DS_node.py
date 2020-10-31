#! /usr/bin/env python
from __future__ import print_function
import rospy, math, sys
from cartHBSModulation_class import CartesianMotionControl_DSModulation
import numpy as np
import numpy.linalg as LA
from numpy import random as np_random

# This is for modulation and visualization
from matplotlib import pyplot as plt
from obstacles import GammaRectangle2D, GammaRectangle3D
from modulation_ros import linear_controller, modulation_HBS


# DS Modulation Libraries with Gamma Function
sys.path.append("./dynamical_system_modulation_svm/")
import learn_gamma_fn
import test_modulation_svm

# Fit Gamma Function
grid_size = 50
X, Y, c_labels = test_modulation_svm.create_franka_dataset(dimension=3, grid_size=grid_size, plot_training_data=0)      
gamma_svm      = 20
c_svm          = 20
grid_limits_x  = [0.1, 1.0]
grid_limits_y  = [-0.8, 0.8]
grid_limits_z  = [0.55, 1.1]
learned_gamma  = learn_gamma_fn.create_obstacles_from_data(data=X, label=Y, 
    plot_raw_data=False,  gamma_svm=gamma_svm, c_svm=c_svm, cluster_labels = c_labels)

# Global Variable
PI = math.pi

if __name__ == '__main__':

    rospy.init_node('cartesianHBSCollisionControl_DS')    

    ####################################################
    #############    Parsing Parameters    #############
    ####################################################
    
    ####### DS variables #######
    DS_type    = rospy.get_param('~ds_type', 1)  # 1: Joint-space DS with joint-space target (JJ-DS)
                                                 # 2: Joint-space DS with task-space target  (JT-DS)
    
    #  Selected goal in joint space #
    goal           = rospy.get_param('~goal', 1)   
    ctrl_orient    = rospy.get_param('~ctrl_orient', 0)   
    draw_DS        = rospy.get_param('~draw_DS', 0)   
    do_streamlines = rospy.get_param('~stream', 1)   

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

    if draw_DS:        
        ###########################################################        
        ''' -- Visualize expected DS vector field in 2D Slice -- '''
        ###########################################################        


    ##############################################
    ''' -- Generate Real 3D Gamma Functions -- '''
    ##############################################
    margin_offset = 1.5
    # Define Reference Points
    ref_point1 =   np.array([0.625, 0.0, 0.60])

    # Merged/T-shape
    ref_point2 =   np.array([0.625, 0.0, 1.025])
    ref_point3  =  np.array([0.625, 0.0, 1.025])

    # Independent Reference Points
    # ref_point2 =   np.array([0.625, 0.0, 1.03])
    # ref_point3  =  np.array([0.625, 0.0, 0.825])    

    # Define Gammas
    gamma1   = GammaRectangle3D(np.array(1.60),  np.array(0.05), np.array(0.2),    np.array([0.625, 0.0, 0.6]),   ref_point1, margin_offset)
    gamma2   = GammaRectangle3D(np.array(0.80),  np.array(0.075),  np.array(0.25),  np.array([0.625, 0.0, 1.045]), ref_point2, margin_offset)
    gamma3   = GammaRectangle3D(np.array(0.075),  np.array(0.375),   np.array(0.25), np.array([0.625, 0.0, 0.825]), ref_point3, margin_offset)
    # gammas = [gamma1, gamma2, gamma3]
    gammas = [gamma2]

    # DS system matrix, gains for each task-space error    
    # A_p = [[2.0, 0, 0], 
    #        [0, 2.0, 0],
    #        [0, 0, 2.0]]

    A_p = [[0.5, 0, 0], 
           [0, 0.5, 0],
           [0, 0, 0.5]]

    A_o = [[3.0, 0, 0], 
           [0, 3.0, 0],
           [0, 0, 3.0]]           

    # Threshold for stopping DS     
    epsilon = 0.025
         
    ####### Motion Control Variables #######
    ctrl_rate   = 150 # 150hz
    
    ####### Initialize Class #######
    # Add the gamma functionS here
    cartVelocityController = CartesianMotionControl_DSModulation(DS_type, A_p, A_o, DS_attractor, ctrl_rate, epsilon, ctrl_orient, gammas)

    ####### Run Control #######
    cartVelocityController.run()

