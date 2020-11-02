#!/usr/bin/env python
# PointCloud2 color cube
# https://answers.ros.org/question/289576/understanding-the-bytes-in-a-pcl2-message/
import sys, rospy
import struct
from sensor_msgs import point_cloud2
# from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np


# DS Modulation Libraries with Gamma Function
# This is ugly but can update later when we make the './dynamical_system_modulation_svm' a package that can be installed
sys.path.append("/home/nbfigueroa/code/bayes-probe-robotics/dynamical_system_modulation_svm")

import learn_gamma_fn
import test_modulation_svm
import modulation_svm
import pickle


epsilon        = sys.float_info.epsilon
grid_size      = 30
grid_limits_x  = [0.1, 1.0]
grid_limits_y  = [-0.8, 0.8]
grid_limits_z  = [0.55, 1.1]
re_learn       = 0

ds_target = []
ee_position = []


def create_points_gamma():
  # CREATE POINT CLOUD TO VISUALIZE GAMMA VALUES!!
  grid_size = 35
  points = []   
  use_gamma = 1
  for k in np.linspace(grid_limits_z[0], grid_limits_z[1], grid_size):  
      for j in np.linspace(grid_limits_y[0], grid_limits_y[1], grid_size):
        for i in np.linspace(grid_limits_x[0], grid_limits_x[1], grid_size):
          x = float(i) 
          y = float(j) 
          z = float(k)

          # Free space
          r = int(0.0 * 255.0)
          g = int(1.0 * 255.0)
          b = int(0.0 * 255.0)
          a = int(0.001 * 255.0)
          obstacle = 0

          # -- Fill in here the colors with gamma function -- #
          if use_gamma:

            # Define Gammas
            classifier        = learned_gamma['classifier']
            max_dist          = learned_gamma['max_dist']
            reference_points  = learned_gamma['reference_points']
            x_eval = np.array([x, y, z])          
            gamma_val  = learn_gamma_fn.get_gamma(x_eval, classifier, max_dist, reference_points, dimension=3)

            print("Gamma vals:", gamma_val)
            if gamma_val < 1:
                r = int(1.0* 255.0)
                g = int(0.0* 255.0)              
                a = int(0.075* 255.0)
                obstacle = 1
            else:
                r = int(0.05 * 255.0)
                g = int(min(1, gamma_val/10) * 255.0)  
                b = int(0.0 * 255.0)  
                a = int(0.075* 255.0)

          else:
            # -- Fill in here the colors with table description -- #
            print("Using geometric table description")
            # The table Top
            if (z < 0.625):
                r = int(1.0* 255.0)
                g = int(0.0* 255.0)              
                a = int(0.075* 255.0)

            # The vertical wall
            if (x>= 0.3):
               if (y>=-0.04 and y<=0.04): # Adding 2cm no the sides (to account for gripper)
                if (z >= 0.625 and z <= 1.025):
                  r = int(1.0* 255.0)
                  g = int(0.0* 255.0)
                  a = int(0.075* 255.0)

            # The horizontal wall
            if (x>= 0.3):
              if (y>=-0.45 and y<=0.45): 
                  if (z >= 0.975 and z <= 1.065): 
                    r = int(1.0* 255.0)
                    g = int(0.0* 255.0)
                    a = int(0.075* 255.0)


          rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
          pt = [x, y, z, rgb]
          if obstacle:
            points.append(pt)

  return points

def get_DS_target(msg):  
  # get position  
  x = msg.point.x
  y = msg.point.y
  z = msg.point.z
  global ds_target
  ds_target = np.array ([x,y,z])
  rospy.loginfo('DS target: {}'.format(ds_target))
  # recieved_target = True

def get_ee_position(msg):
  # get position  
  x = msg.point.x
  y = msg.point.y
  z = msg.point.z  
  global ee_position 
  ee_position = np.array ([x,y,z])
  rospy.loginfo('EE position: {}'.format(ee_position))
  # recieved_position = True


# --- ROS NODE INIT, SUBSCRIBERS AND PUBLISHERS --- #
rospy.init_node("gammaSVM_evaluation_node")
rospy.Subscriber("DS_target", PointStamped, get_DS_target)
rospy.Subscriber("curr_ee_pos", PointStamped, get_ee_position)
pub_gamma  = rospy.Publisher("gamma_values", PointCloud2, queue_size=2)
pub_fw_int = rospy.Publisher("DS_path", Path, queue_size = 2)
# pub_fw_int = rospy.Publisher("/DS/forward_integration", PointCloud2, queue_size=2)

if re_learn:
    # Create Environment Dataset and Learn Gamma!    
    X, Y, c_labels = test_modulation_svm.create_franka_dataset(dimension=3, grid_size=grid_size, plot_training_data=0)      
    gamma_svm      = 20
    c_svm          = 20
    learned_gamma  = learn_gamma_fn.create_obstacles_from_data(data=X, label=Y, 
        plot_raw_data=False,  gamma_svm=gamma_svm, c_svm=c_svm, cluster_labels = c_labels)
else:
    # Load Pre-Learned Model
    learned_gamma, gamma_svm, c_svm = pickle.load(open("/home/nbfigueroa/code/bayes-probe-robotics/dynamical_system_modulation_svm/models/gammaSVM_frankaROCUS.pkl", 'rb'))

points = create_points_gamma()
fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          # PointField('rgb', 12, PointField.UINT32, 1),
          PointField('rgba', 12, PointField.UINT32, 1),
          ]
header = Header()
header.frame_id = "world"
pc2    = point_cloud2.create_cloud(header, fields, points)

i = 0 
ee_first = 0
while not rospy.is_shutdown():
    pc2.header.stamp = rospy.Time.now()
    pub_gamma.publish(pc2)
    
    # # Integrate trajectories from initial point
    # if (i > 10):
    #   ee_first = 1


    # if ee_first and i < 30 and (not ee_position ==[]) :
    #   rospy.loginfo('INTEGRATIIING')

    #   x_traj, x_dot_traj = modulation_svm.forward_integrate_singleGamma_HBS(ee_position, ds_target, learned_gamma, dt = 0.01, eps=0.03, max_N = 10000)
    #   path_shape = x_traj.shape
    #   rospy.loginfo("Length of plan {}".format(path_shape))

    #   if (i > 15):
    #     msg = Path()
    #     msg.header.frame_id = "/world"
    #     msg.header.stamp = rospy.Time.now()
    #     rospy.loginfo("Length of plan {}".format(path_shape))
    #     for ii in range(path_shape[0]):
    #         pose = PoseStamped()
    #         pose.pose.position.x = x_traj[ii,0]
    #         pose.pose.position.y = x_traj[ii,1]
    #         pose.pose.position.z = x_traj[ii,2]
    #         msg.poses.append(pose)
    #         rospy.loginfo("Publishing Plan...")
    #         pub_fw_int.publish(msg) 

    # pub_path.publish()
    rospy.sleep(0.5)
    i += 1



# get orientationn and convert quternion to euler (roll pitch yaw)
  # quaternion = (
  #   msg.pose.orientation.x,
  #   msg.pose.orientation.y,
  #   msg.pose.orientation.z,
  #   msg.pose.orientation.w)
  # euler = tf.transformations.euler_from_quaternion(quaternion)
  # yaw = math.degrees(euler[2])