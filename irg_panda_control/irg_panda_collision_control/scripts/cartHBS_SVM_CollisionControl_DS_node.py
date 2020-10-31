#! /usr/bin/env python
from __future__ import print_function
import rospy, math, sys
from cartHBSModulation_SVM_class import CartesianMotionControl_DSModulation
import numpy as np
import numpy.linalg as LA
from numpy import random as np_random

# DS Modulation Libraries with Gamma Function
sys.path.append("./dynamical_system_modulation_svm/")
import learn_gamma_fn
import modulation_svm
import test_modulation_svm
import pickle

# To Visualize Integrated Trajectory
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped, PoseStamped
from franka_core_msgs.msg import EndPointState

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

    rospy.init_node('cartesianHBSCollisionControl_DS')    

    ####################################################
    #############    Parsing Parameters    #############
    ####################################################
    
    ####### DS variables #######
    DS_type    = rospy.get_param('~ds_type', 1)  # 1: Joint-space DS with joint-space target (JJ-DS)
                                                 # 2: Joint-space DS with task-space target  (JT-DS)
    
    #  Selected goal in joint space #
    goal           = rospy.get_param('~goal', 6)   
    ctrl_orient    = rospy.get_param('~ctrl_orient', 0)   
    draw_DS        = rospy.get_param('~draw_DS', 0)   
    do_streamline  = rospy.get_param('~stream', 1)   

    # control for position + orientation
    # FORMAT:
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
        x,y,z = test_modulation_svm.rand_target_loc()
        if y < 0: 
            DS_attractor   = [x, y, z, 0.683, 0.082, 0.679, -0.256]   
        else:
            DS_attractor   = [x, y, z, 0.878, 0.044, 0.442, 0.181]   

    ##############################################
    ''' -- Generate Real 3D Gamma Function -- '''
    ##############################################
    re_learn = 0

    if re_learn:
        # Create Environment Dataset and Learn Gamma!
        grid_size = 50
        X, Y, c_labels = test_modulation_svm.create_franka_dataset(dimension=3, grid_size=grid_size, plot_training_data=0)      
        gamma_svm      = 20
        c_svm          = 20
        grid_limits_x  = [0.1, 1.0]
        grid_limits_y  = [-0.8, 0.8]
        grid_limits_z  = [0.55, 1.1]
        print("Learning Gamma Function")
        learned_gamma  = learn_gamma_fn.create_obstacles_from_data(data=X, label=Y, 
            plot_raw_data=False,  gamma_svm=gamma_svm, c_svm=c_svm, cluster_labels = c_labels)
        print("DONE.")
    else:
        # Load Pre-Learned Model
        # learned_gamma, gamma_svm, c_svm = pickle.load(open("./dynamical_system_modulation_svm/models/gammaSVM_frankaROCUS.pkl", 'rb'))
        learned_gamma, gamma_svm, c_svm = pickle.load(open("./dynamical_system_modulation_svm/models/gammaSVM_frankaROCUS_bounded.pkl", 'rb'))


    if do_streamline:
        ctrl_rate    = 1000
        rate         = rospy.Rate(ctrl_rate)
        rospy.loginfo('Getting current robot state')  
        pub_fw_int  = rospy.Publisher("DS_path", Path, queue_size = 2)
        # Spin once to update robot state
        ee_msg      = rospy.wait_for_message('/panda_simulator/custom_franka_state_controller/tip_state', EndPointState)
        ee_position = get_endpoint_state(ee_msg)
        rospy.loginfo('Doing Forward Integration')
        x_traj, x_dot_traj = modulation_svm.forward_integrate_singleGamma_HBS(ee_position, DS_attractor[0:3], learned_gamma, dt = 0.05, eps=0.03, max_N = 10000)
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

    # DS system matrix, gains for each task-space error    
    # A_p = [[2.0, 0, 0], 
    #        [0, 2.0, 0],
    #        [0, 0, 2.0]]

    A_p = [[1.0, 0.0, 0.0], 
           [0.0, 1.0, 0.0],
           [0.0, 0.0, 1.0]]

    A_o = [[3.0, 0, 0], 
           [0, 3.0, 0],
           [0, 0, 3.0]]           

    # Threshold for stopping DS     
    epsilon = 0.025
         
    ####### Motion Control Variables #######
    ctrl_rate   = 150 # 150hz

    #################################################
    ''' -- Run DS Modulation + PD-Control Node -- '''
    #################################################

    ####### Initialize Class #######
    print("Starting Control Node")
    # Add the gamma functionS here
    cartVelocityController = CartesianMotionControl_DSModulation(DS_type, A_p, A_o, DS_attractor, ctrl_rate, epsilon, ctrl_orient, learned_gamma)

    ####### Run Control #######
    cartVelocityController.run()

