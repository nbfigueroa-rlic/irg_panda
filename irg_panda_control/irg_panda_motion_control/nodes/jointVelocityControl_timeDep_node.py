#! /usr/bin/env python
from __future__ import print_function
import rospy, math, sys
from jointMotionControl.timeDep_class import JointMotionControl_TimeDependent

# Global Variable
PI = math.pi

if __name__ == '__main__':

    rospy.init_node('jointVelocityControl_timeDep')    

    ####################################################
    #############    Parsing Parameters    #############
    ####################################################
    
    ####### Selected goal in joint space #######
    goal       = rospy.get_param('~goal', 1)       
    if goal == 1:
        # Candle Joint configuration 
        trajGen_goal   = [0, 0, 0, 0, 0, 0]
    elif goal == 2:    
        # Joint configuration for task execution        
        trajGen_goal   = [0, -0.10000000149011612, 1.8849549293518066, 0.00019369914662092924, 1.3799999952316284, 0]        
    elif goal == 3: 
        # Out-of-workspace (right-side)
        trajGen_goal   = [-1.0048859119415283, -0.10332348942756653, 1.675516128540039, 0.27925267815589905, 0.41887903213500977, 0.0]
    elif goal == 4: 
        # Out-of-workspace (left-side)
        trajGen_goal   = [1.5917402505874634, 0.5682792067527771, 1.3089966773986816, 0.27925267815589905, -0.6282899379730225, 0.0]


    ####### Type of Motion Control Law #######
    ctrl_type  = rospy.get_param('~ctrl_type',4)   # 1: ff (Openloop velocity)
                                                   # 2: fb (P-control)
                                                   # 3: fb (PI-control) 
                                                   # 4: ff+fb (Openloop velocity + PI-control) --> Recommended, but be careful with gain! Test in sim first!

    ####### Trajectory Generation Variables #######
    trajGen_Tf      = rospy.get_param('~trajGen_Tf', 3)
    trajGen_method  = rospy.get_param('~trajGen_method', 5)
    # See JointTrajectoryGenerator in modern_robotics class #               

    ####### Command Type (i.e. joint-velocities or joint-positions) #######
    cmd_type = 1   # 1: Joint velocities commanded to the robot
                   # 2: Joint positions  commanded to the robot

    ####### Motion Control Variables #######
    ctrl_rate      = 280 # 150hz

    ###  Gain for P-Control, used to automatically set K_i (critically damped error dynamics) ###
    # For P-Control use 'high' gain; 
    # i.e. K_p > 10 (not so jerky motion w/3s trajectory)
    Kp_default = 10                                                                                                                                      
    
    # For PI-Control use moderate gain; 
    # i.e.  K_p ~ 7.5 (less jerky but with a bit of an overshoot when reaching the target)
    # K_p ~ 5 gives smooth motion, no overshoot, but less accurate at target
    if ctrl_type == 3:
        Kp_default = 7.5

    # For FF+PI-Control use lower gain; 
    # i.e. K_p > 2.5 (better!)
    if ctrl_type == 4:
        Kp_default = 2.5
                        
    #### NOTE: ALL OF THESE VALUES ARE TUNED FOR A 3S trajectory! ####                    
    K_p   = rospy.get_param('~K_p', Kp_default) 

    ####### Initialize Class #######
    jointVelocityController = JointMotionControl_TimeDependent(trajGen_goal, trajGen_Tf, trajGen_method, ctrl_rate, cmd_type, ctrl_type, K_p)

    ####### Run Control #######
    jointVelocityController.run()
