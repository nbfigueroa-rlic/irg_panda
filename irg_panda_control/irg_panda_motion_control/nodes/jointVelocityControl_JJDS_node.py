#! /usr/bin/env python
from __future__ import print_function
import rospy, math, sys
from jointMotionControl.stateDep_class import JointMotionControl_StateDependent

# Global Variable
PI = math.pi

if __name__ == '__main__':

    rospy.init_node('jointVelocityControl_JJDS')    
    rospy.wait_for_service('/controller_manager/list_controllers')
    rospy.loginfo("Starting node...")
    rospy.sleep(1)

    ####################################################
    #############    Parsing Parameters    #############
    ####################################################
    
    ####### DS variables #######
    DS_type    = rospy.get_param('~ds_type', 1)  # 1: Joint-space DS with joint-space target (JJ-DS)
                                                 # 2: Joint-space DS with task-space target  (JT-DS)
    #  Selected goal in joint space #
    goal       = rospy.get_param('~goal', 1)       
    if goal == 1:
        # Candle Joint configuration 
        DS_attractor   = [0, 0, 0, 0, 0, 0, 0]
    elif goal == 2:    
        # Joint configuration for task execution        
        DS_attractor   = [0.0, -0.3, 0.0, -2.0, 0.0, 2.0, 0.785]       
    elif goal == 3: 
        # Out-of-workspace (right-side)
        DS_attractor   = [-1.0048859119415283, -0.10332348942756653, 1.675516128540039, 0.27925267815589905, 0.41887903213500977, 0.0]
    elif goal == 4: 
        # Out-of-workspace (left-side)
        DS_attractor   = [1.5917402505874634, 0.5682792067527771, 1.3089966773986816, 0.27925267815589905, -0.6282899379730225, 0.0]


    # DS system matrix, gains for each joint error    
    A = [[3, 0, 0, 0, 0, 0, 0], 
         [0, 5, 0, 0, 0, 0, 0],
         [0, 0, 6, 0, 0, 0, 0],
         [0, 0, 0, 6, 0, 0, 0],
         [0, 0, 0, 0, 6, 0, 0],
         [0, 0, 0, 0, 0, 6, 0] ,
         [0, 0, 0, 0, 0, 0, 6]]

    # Threshold for stopping DS     
    epsilon = 0.025
         
    ####### Command Type (i.e. joint-velocities or joint-positions) #######
    cmd_type = 1   # 1: Joint velocities commanded to the robot
                   # 2: Joint positions  commanded to the robot

    ####### Motion Control Variables #######
    ctrl_rate      = 1000 # 280hz = 3.555 ms
    
    ####### Initialize Class #######
    jointVelocityController = JointMotionControl_StateDependent(DS_type, A, DS_attractor, ctrl_rate, cmd_type, epsilon)

    ####### Run Control #######
    jointVelocityController.run()
