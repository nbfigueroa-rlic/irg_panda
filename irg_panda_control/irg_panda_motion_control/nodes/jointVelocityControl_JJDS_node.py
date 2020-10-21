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
        # Joint configuration for task execution
        DS_attractor   = [-0.017792060227770554, -0.7601235411041661, 0.019782607023391807, -2.342050140544315, 0.029840531355804868, 1.5411935298621688, 0.7534486589746342]
    
    elif goal == 2:    
        DS_attractor   = [-0.2503045358116065, 0.18778477483718525, -0.7758685631151208, -1.946400309929718, 0.13797851088629276, 2.0297926843732874, -0.3084721190836994]
        
    elif goal == 3: 
        # Inside shelf (right-side)
        DS_attractor = [-1.0549702240639425, 1.5264971713923776, 1.168276536618877, -0.8192400136671232, 2.747412073267812, 3.7525061076536206, -2.8973000113743783]
        
    elif goal == 4: 
        # Out-of-workspace (left-side)
        DS_attractor   = [0.032, -0.3, 0.0, -2.0, 0.0, 2.0, PI/4.0]


    # DS system matrix, gains for each joint error    
    A = [[5, 0, 0, 0, 0, 0, 0], 
         [0, 5, 0, 0, 0, 0, 0],
         [0, 0, 5, 0, 0, 0, 0],
         [0, 0, 0, 5, 0, 0, 0],
         [0, 0, 0, 0, 5, 0, 0],
         [0, 0, 0, 0, 0, 5, 0] ,
         [0, 0, 0, 0, 0, 0, 5]]

    # Threshold for stopping DS     
    epsilon = 0.075
         
    ####### Command Type (i.e. joint-velocities or joint-positions) #######
    cmd_type = 1   # 1: Joint velocities commanded to the robot
                   # 2: Joint positions  commanded to the robot

    ####### Motion Control Variables #######
    ctrl_rate      = 100 # 280hz = 3.555 ms
    
    ####### Initialize Class #######
    jointVelocityController = JointMotionControl_StateDependent(DS_type, A, DS_attractor, ctrl_rate, cmd_type, epsilon)

    ####### Run Control #######
    jointVelocityController.run()
