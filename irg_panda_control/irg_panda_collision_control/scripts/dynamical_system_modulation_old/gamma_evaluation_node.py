#!/usr/bin/env python
# PointCloud2 color cube
# https://answers.ros.org/question/289576/understanding-the-bytes-in-a-pcl2-message/
import sys, rospy
import struct

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

import numpy as np
from obstacles import GammaRectangle2D, GammaRectangle3D

rospy.init_node("gamma_evaluation_node")
pub = rospy.Publisher("gamma_values", PointCloud2, queue_size=2)

grid_size = 35
points = []   

use_gamma = 1
epsilon = sys.float_info.epsilon


for k in np.linspace(0.55, 1.1, grid_size):  
    for j in np.linspace(-0.8, 0.8, grid_size):
      for i in np.linspace(0.1, 1.0, grid_size):
        x = float(i) 
        y = float(j) 
        z = float(k)

        # Free space
        r = int(0.0 * 255.0)
        g = int(1.0 * 255.0)
        b = int(0.0 * 255.0)
        a = int(0.001 * 255.0)

        # -- Fill in here the colors with gamma function -- #
        if use_gamma:
          margin_offset = 1.5
          # Define Reference Points
          ref_point1 =   np.array([0.625, 0.0, 0.60])

          # Merged/T-shape
          ref_point2 =   np.array([0.625, 0.0, 1.025])
          ref_point3  =  np.array([0.625, 0.0, 1.025])

          # Independent Reference Points
          # ref_point2 =   np.array([0.625, 0.0, 1.03])
          # ref_point3  =  np.array([0.625, 0.0, 0.825])    

          # Define Gammas
          gamma1   = GammaRectangle3D(np.array(1.60),  np.array(0.05), np.array(0.2),    np.array([0.625, 0.0, 0.6]),   ref_point1, margin_offset)
          gamma2   = GammaRectangle3D(np.array(0.80),  np.array(0.075),  np.array(0.25),  np.array([0.625, 0.0, 1.045]), ref_point2, margin_offset)
          gamma3   = GammaRectangle3D(np.array(0.075),  np.array(0.375),   np.array(0.25), np.array([0.625, 0.0, 0.825]), ref_point3, margin_offset)
          gammas = [gamma1, gamma2, gamma3]
          x_eval = np.array([x, y, z])          
          # print("Gamma vals:", gamma_vals)
          if min([gamma(x_eval) for gamma in gammas]) < 1:
              r = int(1.0* 255.0)
              g = int(0.0* 255.0)              
              a = int(0.075* 255.0)
          else:
              gamma_vals = np.stack([gamma(x_eval) for gamma in gammas])  
              min_gamma = min(gamma_vals)
              max_gamma = max(gamma_vals)
              r = int(0.05 * 255.0)
              g = int(min(1, min_gamma/max_gamma) * 255.0)  
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


        # print r, g, b, a
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        # print hex(rgb)
        pt = [x, y, z, rgb]
        points.append(pt)

fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          # PointField('rgb', 12, PointField.UINT32, 1),
          PointField('rgba', 12, PointField.UINT32, 1),
          ]

print("Done creating point cloud")
header = Header()
header.frame_id = "world"
pc2 = point_cloud2.create_cloud(header, fields, points)

while not rospy.is_shutdown():
    pc2.header.stamp = rospy.Time.now()
    pub.publish(pc2)
    rospy.sleep(1.0)
