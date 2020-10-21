#! /usr/bin/env python
from __future__ import print_function
import rospy, math, sys
from jointMotionControl.stateDep_class import JointMotionControl_StateDependent

# Global Variable
PI = math.pi

if __name__ == '__main__':

    rospy.init_node('jointPositionControl_JJDS')    
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
        # DS_attractor   = [-0.017792060227770554, -0.7601235411041661, 0.019782607023391807, -2.342050140544315, 0.029840531355804868, 1.5411935298621688, 0.7534486589746342]
        DS_attractor   = [0.0, -0.3, 0.0, -2.0, 0.0, 2.0, PI/4.0]
    elif goal == 2:    
        # Outside shelf (right-side)
        DS_attractor   = [-0.49041676056762995, 0.7766592902718523, -0.7734337363858659, -1.4823211274344121, 0.6100456652624864, 1.9938822682398296, -0.647085016124489]

    elif goal == 3: 
        # Inside shelf (right-side)
        DS_attractor = [-1.1125535140101084, 1.3894461610296673, 1.1576712539494416, -1.4770406551167259, 1.3767816205572458, 2.59394752449543, -1.6974494234802124]        

    elif goal == 4: 
        # Outside shelf (left-side)
        DS_attractor   = [0.2963225263724816, 0.4153249148679894, 0.9195334805130182, -1.801444127631573, -0.2435031473087399, 2.0238766403464794, 2.047426326036671]

    elif goal == 5: 
        # Inside shelf (left-side)
        DS_attractor   = [1.4140998302647318, 1.2103373992676127, -1.1311743519704436, -1.8853877571783464, -0.15587487157180568, 3.096325509881675, 1.8390013243372465]

    # DS system matrix, gains for each joint error    
    A = [[7.5, 0, 0, 0, 0, 0, 0], 
         [0, 7.5, 0, 0, 0, 0, 0],
         [0, 0, 7.5, 0, 0, 0, 0],
         [0, 0, 0, 7.5, 0, 0, 0],
         [0, 0, 0, 0, 7.5, 0, 0],
         [0, 0, 0, 0, 0, 7.5, 0] ,
         [0, 0, 0, 0, 0, 0, 7.5]]

    # Threshold for stopping DS     
    epsilon = 0.075
         
    ####### Command Type (i.e. joint-velocities or joint-positions) #######
    cmd_type = 2   # 1: Joint velocities commanded to the robot
                   # 2: Joint positions  commanded to the robot

    ####### Motion Control Variables #######
    ctrl_rate      = 100 # 280hz = 3.555 ms
    
    ####### Initialize Class #######
    jointPositionController = JointMotionControl_StateDependent(DS_type, A, DS_attractor, ctrl_rate, cmd_type, epsilon)

    ####### Run Control #######
    jointPositionController.run()
