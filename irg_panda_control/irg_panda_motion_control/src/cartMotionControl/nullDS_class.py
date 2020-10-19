from __future__ import print_function
import rospy, math, time
from geometry_msgs.msg import Twist
import numpy as np


class CartesianMotionControl_NullDS(object):
    """
        This class sends desired null DS to twist command, i.e. 0 linear or angular velocity
    """
    def __init__(self,ctrl_rate = 150):

        # ---Publishes twist command to filter node
        self._pub_twist   = rospy.Publisher('/UR10arm/desired_twist', Twist, queue_size=10)        
        
        # Robot commands        
        self.twist_msg           = Twist()
        self.twist_msg.linear.x  = 0
        self.twist_msg.linear.y  = 0
        self.twist_msg.linear.z  = 0
        self.twist_msg.angular.x = 0
        self.twist_msg.angular.y = 0
        self.twist_msg.angular.z = 0        

        # Variable for control loop
        self.rate          = rospy.Rate(ctrl_rate)

    def _compute_desired_velocities(self):
        """
            Compute desired joint velocities from state-dependent DS
        """        
    
        # Compute Desired Linear Velocity
        lin_vel = np.array([0,0,0])
    
        # Compute Desired Angular Velocity
        ang_vel = np.array([0,0,0])

        return lin_vel, ang_vel
   
    def _publish_desired_twist(self, lin_vel, ang_vel):
        """
            Convert numpy arrays to Twist-msg and publish
        """       
    
        # Populate twist message
        self.twist_msg.linear.x  = lin_vel[0]
        self.twist_msg.linear.y  = lin_vel[1]
        self.twist_msg.linear.z  = lin_vel[2]
        self.twist_msg.angular.x = ang_vel[0]
        self.twist_msg.angular.y = ang_vel[1]
        self.twist_msg.angular.z = ang_vel[2]     

        # Publish command to low-level controller
        self._pub_twist.publish(self.twist_msg)

        # Uncomment if you want to see the desired joint velocities sent to the robot        
        # rospy.loginfo('Desired twist: {}'.format(self.twist_msg))

    def run(self): 
        """
            The actual control-loop that sends commands (position/velocity) to the robot arm
        """     
        while not rospy.is_shutdown():
        
            # Compute desired velocities from DS
            lin_vel, ang_vel = self._compute_desired_velocities()              
            
            # Publish desired twist
            self._publish_desired_twist(lin_vel, ang_vel)     

            # Control-loop rate
            self.rate.sleep()