#! /usr/bin/env python
from __future__ import print_function
import rospy, math, sys
from jointMotionControl.jointMotionControl_stateDep_class import JointMotionControl_StateDependent

# Global Variable
PI = math.pi

if __name__ == '__main__':

    rospy.init_node('jointVelocityControl_JTDS')    

    ####################################################
    #############    Parsing Parameters    #############
    ####################################################
    
    ####### DS variables #######
    DS_type    = rospy.get_param('~ds_type', 2)  # 1: Joint-space DS with joint-space target (JJ-DS)
                                                 # 2: Joint-space DS with task-space target  (JT-DS)
    #  Selected goal in joint space #
    goal       = rospy.get_param('~goal', 1)       

    # control for position + orientation
    # DS_attractor= [x, y, z, q_x, q_y, q_z, q_w]
    if goal == 1:
        ### Inside-Workspace (over-table) ###        
        DS_attractor   = [1.00, 0.164, 0.714, 0.269, 0.662, -0.276, 0.643] 
    elif goal == 2:    
        ### Out-of-Workspace (left-side) ###                
        DS_attractor   = [0.618, 0.673, 1.266, 0.153, 0.378, 0.317, 0.856]

        ### In-Workspace (front-bottom-left) ###                
        # DS_attractor   = [1.00, 0.664, 0.714, 0.269, 0.662, -0.276, 0.643] 

    elif goal == 3: 
        ### EE top-right of workspace ###
        DS_attractor   = [0.495, -0.688, 1.128, 0.406, -0.027, -0.713, 0.570]
                
        ### In-Workspace (front-bottom-right) ###                
        # DS_attractor   = [1.00, -0.464, 0.714, 0.269, 0.662, -0.276, 0.643] 

    elif goal == 4:
        ### In-Workspace (front-top) ###                
        DS_attractor   = [1.001, 0.164, 1.055, 0.267, 0.662, -0.274, 0.645]


    # DS system matrix, gains for each joint error    
    # A = [[1, 0, 0, 0, 0, 0], 
    #      [0, 1, 0, 0, 0, 0],
    #      [0, 0, 1, 0, 0, 0],
    #      [0, 0, 0, 1, 0, 0],
    #      [0, 0, 0, 0, 1, 0],
    #      [0, 0, 0, 0, 0, 1]]
    
    A = [[1, 0, 0, 0, 0, 0], 
         [0, 2, 0, 0, 0, 0],
         [0, 0, 2, 0, 0, 0],
         [0, 0, 0, 2, 0, 0],
         [0, 0, 0, 0, 2, 0],
         [0, 0, 0, 0, 0, 2]]

    # Threshold for stopping DS     
    epsilon = 0.02
         
    ####### Command Type (i.e. joint-velocities or joint-positions) #######
    cmd_type = 1   # 1: Joint velocities commanded to the robot
                   # 2: Joint positions  commanded to the robot

    ####### Motion Control Variables #######
    ctrl_rate      = 150 # 150hz
    
    ####### Initialize Class #######
    jointVelocityController = JointMotionControl_StateDependent(DS_type, A, DS_attractor, ctrl_rate, cmd_type, epsilon)

    ####### Run Control #######
    jointVelocityController.run()
