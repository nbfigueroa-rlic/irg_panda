#! /usr/bin/env python
from __future__ import print_function
import rospy, math, sys
from cartMotionControl.stateDep_class import CartesianMotionControl_StateDependent

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
    if goal == 1:
        ### Inside-Workspace (over-table) ###        
        DS_attractor   =    [0.450, 0.000, 0.445, 0.000, 1.000, 0.000, -0.001]

    elif goal == 2:
        ### Inside-Workspace (over-table) ###        
        DS_attractor   =    [0.450, 0.000, 0.045, 0.000, 1.000, 0.000, -0.001]

    elif goal == 3:    
        ### In-Workspace (front-bottom-left) ###  
        DS_attractor   =    [0.450, -0.45, 0.045, 0.000, 1.000, 0.000, -0.001]

    elif goal == 4:                 
        ### In-Workspace (front-bottom-left) ###  
        DS_attractor   =    [0.450, 0.45, 0.045, 0.000, 1.000, 0.000, -0.001]
        # ### In-Workspace (front-bottom-right) ###                
        # DS_attractor   = [0.022, 0.815, 0.490, -0.315, 0.503, 0.598, 0.538] 


    # DS system matrix, gains for each joint error    
    A_p = [[1.75, 0, 0], 
           [0, 1.75, 0],
           [0, 0, 1.75]]

    # A_o = [[1.0, 0, 0], 
    #        [0, 1.0, 0],
    #        [0, 0, 1.0]]           

    A_o = [[5.0, 0, 0], 
           [0, 5.0, 0],
           [0, 0, 5.0]]           


    # Threshold for stopping DS     
    epsilon = 0.0075
         
    ####### Motion Control Variables #######
    ctrl_rate  = 1000 # 150hz
    
    ####### Initialize Class #######
    cartVelocityController = CartesianMotionControl_StateDependent(DS_type, A_p, A_o, DS_attractor, ctrl_rate, epsilon)

    ####### Run Control #######
    cartVelocityController.run()
