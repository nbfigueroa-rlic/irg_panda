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

def rand_target_loc(np_random):
    '''
    generate random target location
    '''
    # In BULLET's reference frame
    # x = np_random.uniform(low=0.05, high=0.5)
    # if np_random.randint(0, 2) == 0:
    #     x = -x
    # y = np_random.uniform(low=-0.3, high=0.2)

    # reference frames are different in gazebo and pybullet
    y = np_random.uniform(low=0.10, high=0.5)
    if np_random.randint(0, 2) == 0:
        y = -y
    
    x = np_random.uniform(low = 0.3, high = 0.8)
    z = np_random.uniform(low=0.65,  high = 0.9) #Smaller due to 
    return x, y, z

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
    draw_DS        = rospy.get_param('~draw_DS', 1)   
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
        x_target      = np.array([DS_attractor[1], DS_attractor[2]])
        margin_offset = 1.5
        # Define Reference Points
        ref_point1 =   np.array([0.0, 0.60])

        # Merged/T-shape
        ref_point2 =   np.array([0.0, 1.025])
        ref_point3  =  np.array([0.0, 1.025])

        # Independent Reference Points
        # ref_point2 =   np.array([0.0, 1.03])
        # ref_point3  =  np.array([0.0, 0.825])

        # Define Gammas
        gamma1    = GammaRectangle2D(np.array(1.60),  np.array(0.05),  np.array([0.0, 0.6]),  ref_point1, margin_offset)
        gamma2    = GammaRectangle2D(np.array(0.80),  np.array(0.075),  np.array([0.0, 1.045]),  ref_point2, margin_offset)
        gamma3    = GammaRectangle2D(np.array(0.075),  np.array(0.375),  np.array([0.0, 0.825]), ref_point3, margin_offset)
        # gammas_2d = [gamma1, gamma2, gamma3]
        gammas_2d = [gamma2]

        if do_streamlines:
            # Add streamline plot from learned LPV-DS
            fig, ax1 = plt.subplots()        
            # Create figure/environment to draw trajectories on
            fig1, ax1 = plt.subplots()
            ax1.set_xlim(-0.8, 0.8)
            ax1.set_ylim(0.55, 1.25)
            # plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel('$x_1$',fontsize=15)
            plt.ylabel('$x_2$',fontsize=15)
            plt.title('HBS Modulated DS 2D slice (x-axis):',fontsize=15)

            for gamma in gammas_2d:
                gamma.draw()

            grid_size = 50
            Y, X = np.mgrid[0.55:1.25:50j, -0.8:0.8:50j]
            V, U = np.mgrid[0.55:1.25:50j, -0.8:0.8:50j]
            gamma_eval = gamma3
            gamma_val = np.zeros((len(X), len(Y)))
            for i in range(grid_size):
                for j in range(grid_size):
                    x_query    = np.array([X[i,j], Y[i,j]])        
                    orig_ds    = linear_controller(x_query, x_target)
                    mod_x_dot  = modulation_HBS(x_query, orig_ds, gammas_2d)
                    x_dot_norm = mod_x_dot/LA.norm(mod_x_dot) * 0.02
                    U[i,j]     = x_dot_norm[0]
                    V[i,j]     = x_dot_norm[1]

                    gamma_vals     = np.stack([gamma(x_query) for gamma in gammas_2d])
                    gamma_min      = min(gamma_vals)
                    gamma_val[i,j] = gamma_min

            strm      = ax1.streamplot(X, Y, U, V, density = 3.5, linewidth=0.55, color='k')
            levels    = np.array([0, 1])
            # cs0       = ax1.contour(X, Y, gamma_val, levels, origin='lower', colors='k', linewidths=2)
            cs        = ax1.contourf(X, Y, gamma_val, np.arange(-5, 10, 2), cmap=plt.cm.coolwarm, extend='both', alpha=0.8)
            cbar      = plt.colorbar(cs)
            # cbar.add_lines(cs0)

            # Add trajectories used to learn DS
            ax1.plot([ref_point1[0]], [ref_point1[1]], 'y*')
            ax1.plot([ref_point2[0]], [ref_point2[1]], 'y*')
            ax1.plot([ref_point3[0]], [ref_point3[1]], 'y*')
            ax1.plot(x_target[0], x_target[1], 'md', markersize=12, lw=2)
            plt.savefig('../data/stream_lines_HBS.png')
        else: 
            plt.figure()
            for i in np.linspace(-0.8, 0.8, 50):
                for j in np.linspace(0.55, 1.1, 50):                    
                    x = np.array([i, j])
                    if min([gamma(x) for gamma in gammas_2d]) < 1:
                        continue
                    orig_ds = linear_controller(x, x_target)
                    modulated_x_dot = modulation_HBS(x, orig_ds, gammas_2d) * 0.05
                    plt.arrow(i, j, modulated_x_dot[0], modulated_x_dot[1],
                        head_width=0.008, head_length=0.01)

            for gamma in gammas_2d:
                gamma.draw()
            plt.axis([-0.8, 0.8, 0.55, 1.1])
            plt.plot([x_target[0]], [x_target[1]], 'r*')
            plt.plot([ref_point1[0]], [ref_point1[1]], 'y*')
            plt.plot([ref_point2[0]], [ref_point2[1]], 'y*')
            plt.plot([ref_point3[0]], [ref_point3[1]], 'y*')
            plt.savefig('../data/vector_field_HBS.png')

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

