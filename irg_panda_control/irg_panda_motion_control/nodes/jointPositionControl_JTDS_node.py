#! /usr/bin/env python
from __future__ import print_function
import rospy, math, sys
from jointMotionControl.stateDep_class import JointMotionControl_StateDependent

# Global Variable
PI = math.pi

if __name__ == '__main__':

    rospy.init_node('jointPositionControl_JTDS')    
    rospy.wait_for_service('/controller_manager/list_controllers')
    rospy.loginfo("Starting node...")
    rospy.sleep(1)


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
        DS_attractor   = [0.516, -0.000, 1.221, 0.983, -0.000, 0.183, 0.000] 

    elif goal == 2:    
        ### Out-of-Workspace (left-side) ###                
        DS_attractor   = [0.356, -0.605, 0.891, 0.999, 0.028, 0.040, -0.018]

    elif goal == 3: 
        ### EE top-right of workspace ###
        DS_attractor   = [0.738, -0.197, 0.764, 0.683, 0.082, 0.679, -0.256]   
    
    elif goal == 4:
        DS_attractor   = [0.303, 0.547, 0.969, 0.999, -0.017, -0.031, -0.009]

    elif goal == 5:
        DS_attractor   = [0.627, 0.312, 0.763, 0.878, 0.044, 0.442, 0.181]

    # DS system matrix, gains for each joint error    
    A = [[7.5, 0, 0, 0, 0, 0, 0], 
         [0, 7.5, 0, 0, 0, 0, 0],
         [0, 0, 7.5, 0, 0, 0, 0],
         [0, 0, 0, 7.5, 0, 0, 0],
         [0, 0, 0, 0, 10, 0, 0],
         [0, 0, 0, 0, 0, 10, 0] ,
         [0, 0, 0, 0, 0, 0, 10]]

    # Threshold for stopping DS     
    epsilon = 0.02
         
    ####### Command Type (i.e. joint-velocities or joint-positions) #######
    cmd_type = 2   # 1: Joint velocities commanded to the robot
                   # 2: Joint positions  commanded to the robot

    ####### Motion Control Variables #######
    ctrl_rate      = 1000 # 150hz
    
    ####### Initialize Class #######
    jointPositionController = JointMotionControl_StateDependent(DS_type, A, DS_attractor, ctrl_rate, cmd_type, epsilon)

    ####### Run Control #######
    jointPositionController.run()
