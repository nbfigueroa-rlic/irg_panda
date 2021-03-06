#! /usr/bin/env python
from __future__ import print_function
import rospy, math, sys
from cartHBSModulation_SVM_class import CartesianMotionControl_DSModulation
import numpy as np
import numpy.linalg as LA
from numpy import random as np_random

# DS Modulation Libraries with Gamma Function
# This is ugly but can update later when we make the './dynamical_system_modulation_svm' a package that can be installed
sys.path.append("/home/nbfigueroa/code/bayes-probe-robotics/dynamical_system_modulation_svm")

import learn_gamma_fn
import modulation_svm
import test_modulation_svm
import pickle

# To Visualize Integrated Trajectory
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped, PoseStamped
from franka_core_msgs.msg import EndPointState

# To move robot to initial positions!!!
from jointMotionControl.stateDep_class import JointMotionControl_StateDependent
from std_msgs.msg import String


# Global Variable
PI = math.pi

def get_endpoint_state(msg):
    """
        Callback function to get current end-point state
    """
    # Relative pose between world and arm-base
    T_base_rel = np.array([  [1,  0,  0, 0],
                             [0,  1,  0, 0],
                             [0,  0,  1,  0.625],
                              [0, 0, 0, 1]])


    T_ee_panda   = np.array([[1, 0, 0, 0], 
                             [0 , 1 , 0, 0],
                             [0, 0, 1, 0.058],
                             [0, 0, 0, 1]])


    # pose message received is a vectorised column major transformation matrix
    cart_pose_trans_mat = np.asarray(msg.O_T_EE).reshape(4,4,order='F')
    # CARTESIAN_POSE = {
    #     'position': cart_pose_trans_mat[:3,3],
    #     'orientation': quaternion.from_rotation_matrix(cart_pose_trans_mat[:3,:3]) }
    
    ee_pose = np.dot(np.dot(T_base_rel,cart_pose_trans_mat),T_ee_panda)
    ee_pos  = ee_pose[0:3,3]
    return ee_pos
    # If I need Quaternion
    # self.ee_rot  = cart_pose_trans_mat[:3,:3]
    # self.ee_quat = Quaternion(matrix=self.ee_rot)


if __name__ == '__main__':

    rospy.init_node('pyBullet_samplerBridge_node')
    DS_type    = rospy.get_param('~ds_type', 1)
    ctrl_orient    = rospy.get_param('~ctrl_orient', 0)    
    do_streamline  = rospy.get_param('~stream', 0)   


    #################################
    ''' -- Load Gamma Function -- '''
    #################################    
    # learned_gamma, gamma_svm, c_svm = pickle.load(open("./dynamical_system_modulation_svm/models/gammaSVM_frankaROCUS.pkl", 'rb'))
    learned_gamma, gamma_svm, c_svm = pickle.load(open("/home/nbfigueroa/code/bayes-probe-robotics/dynamical_system_modulation_svm/models/gammaSVM_frankaROCUS_bounded.pkl", 'rb'))


    if do_streamline:
        ctrl_rate    = 1000
        rate         = rospy.Rate(ctrl_rate)
        rospy.loginfo('Getting current robot state')  
        pub_fw_int  = rospy.Publisher("DS_path", Path, queue_size = 2)
        # Spin once to update robot state
        ee_msg      = rospy.wait_for_message('/panda_simulator/custom_franka_state_controller/tip_state', EndPointState)
        ee_position = get_endpoint_state(ee_msg)
        rospy.loginfo('Doing Forward Integration')
        x_traj, x_dot_traj = modulation_svm.forward_integrate_singleGamma_HBS(ee_position, DS_attractor[0:3], 
            learned_gamma, dt = 0.05, eps=0.03, max_N = 10000)
        path_shape = x_traj.shape
        rospy.loginfo("Length of plan {}".format(path_shape))
        msg = Path()
        msg.header.frame_id = "/world"
        msg.header.stamp = rospy.Time.now()
        rospy.loginfo("Length of plan {}".format(path_shape))
        for ii in range(path_shape[0]):
          pose = PoseStamped()
          pose.pose.position.x = x_traj[ii, 0]
          pose.pose.position.y = x_traj[ii, 1]
          pose.pose.position.z = x_traj[ii, 2]
          msg.poses.append(pose)
          rospy.loginfo("Publishing Plan...")
          pub_fw_int.publish(msg)
          rate.sleep()         


    ##############################################
    ''' -- DS Parameters -- '''
    ##############################################
    A_p = [[1.0, 0.0, 0.0], 
           [0.0, 1.0, 0.0],
           [0.0, 0.0, 1.0]]

    A_o = [[3.0, 0, 0], 
           [0, 3.0, 0],
           [0, 0, 3.0]]      


    # DS system matrix, gains for each joint error    
    A_q = [[7.5, 0, 0, 0, 0, 0, 0], 
         [0, 7.5, 0, 0, 0, 0, 0],
         [0, 0, 7.5, 0, 0, 0, 0],
         [0, 0, 0, 7.5, 0, 0, 0],
         [0, 0, 0, 0, 7.5, 0, 0],
         [0, 0, 0, 0, 0, 7.5, 0] ,
         [0, 0, 0, 0, 0, 0, 7.5]]                

    # Threshold for stopping DS     
    epsilon_task = 0.03
    epsilon_joint = 0.075 
     
    DS_type_task  = 1 # torque control
    DS_type_joint = 2 # position control     

    ####### Motion Control Variables #######
    ctrl_rate   = 150 # 150hz

    #################################################
    ''' -- Run DS Modulation + PD-Control Node -- '''
    #################################################

    first = 0 
    finished_joint_init = True
    # while not rospy.is_shutdown():
    for i in range(2):
        
        rospy.loginfo("Waiting for NEW target from SAMPLER")

        # Wait for Sampled Target..
        sample_msg      = rospy.wait_for_message('/sampled_target', PointStamped)           
        x = sample_msg.point.y + 0.6
        y = -sample_msg.point.x 
        z = sample_msg.point.z
        DS_attractor_sampled = [x,y,z,0,0,0,1]

        # Get Current EE position
        ee_msg      = rospy.wait_for_message('/panda_simulator/custom_franka_state_controller/tip_state', EndPointState)
        ee_position = get_endpoint_state(ee_msg)

        # Replace this with calls to action server to move robot in joint space
        # if first > 0:
        #     if ee_position[1] < 0:
        #         # Outside shelf (right-side)             
        #         DS_attractor_joint_side   = [-0.49041676056762995, 0.7766592902718523, -0.7734337363858659, -1.4823211274344121, 
        #         0.6100456652624864, 1.9938822682398296, -0.647085016124489]
        #     else:
        #         # Outside shelf (left-side)
        #         DS_attractor_joint_side   = [0.2963225263724816, 0.4153249148679894, 0.9195334805130182, -1.801444127631573, -0.2435031473087399, 
        #         2.0238766403464794, 2.047426326036671]

        #     # ----------- Move robot to right side with joint position trajectory ----------- #
        #     ####### Initialize Class #######
        #     jointPositionController = JointMotionControl_StateDependent(DS_type_joint, A_q, DS_attractor_joint_side, ctrl_rate, 2, epsilon_joint)
        #     ####### Run Control #######
        #     finished_joint = jointPositionController.run()
            
        #     if finished_joint:     
        #         # ----------- Move robot to initial position with collision avoidance! ----------- #
        #         ### Inside-Workspace (over-table) ###        
        #         DS_attractor_task_init   = [0.516, -0.000, 1.221, 0.983, -0.000, 0.183, 0.000] 
        #         # Add the gamma functionS here
        #         cartVelocityController = CartesianMotionControl_DSModulation(DS_type_task, A_p, A_o, DS_attractor_task_init, ctrl_rate, 
        #             epsilon_task, ctrl_orient, learned_gamma, pub_pose = 0)
        #         ####### Run Control #######
        #         finished_cart = cartVelocityController.run()
        

        #     # ----------- Fix initial joint configuration ----------- #
        #     DS_attractor_joint_init   = [0.0, -0.3, 0.0, -2.0, 0.0, 2.0, PI/4.0]
        #     ####### Initialize Class #######
        #     jointPositionController = JointMotionControl_StateDependent(DS_type_joint, A_q, DS_attractor_joint_init, ctrl_rate, 2, epsilon_joint)
        #     ####### Run Control #######
        #     finished_joint_init = jointPositionController.run()


        pub = rospy.Publisher('ready', String, queue_size=10)
        hello_str = "READY"
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rospy.Rate(ctrl_rate).sleep()

        # ----------- RUN OBSTACLE AVOIDANCE TASK WITH SAMPLED TARGET ----------- #
        if finished_joint_init:
            # Add the gamma functionS here
            cartVelocityController = CartesianMotionControl_DSModulation(DS_type_task, A_p, A_o, DS_attractor_sampled, ctrl_rate, epsilon_task, 
                ctrl_orient, learned_gamma)
            ####### Run Control #######
            cartVelocityController.run()

        # Repeat!
        # rospy.Rate(ctrl_rate).sleep()
        first = first + 1