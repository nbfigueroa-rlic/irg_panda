from __future__ import print_function
import rospy, math
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg    import Float64MultiArray
import numpy as np
from   numpy import linalg as LA
import modern_robotics as mr


# Globally defined variables
PI = math.pi
arm_joint_names = [
    'arm_shoulder_pan_joint', 'arm_shoulder_lift_joint', 'arm_elbow_joint',
    'arm_wrist_1_joint', 'arm_wrist_2_joint', 'arm_wrist_3_joint'
]

# Linear and angular velocity limits
vel_limits = [0.15, 0.5]

class CartesianMotionControl_TimeDependent(object):
    """
        This class sends joint velocities/positions to the UR10 by following a time-dependent a straight-line trajectory 
        in joint space that reaches a joint-position goal with different joint velocity control methods
    """
    def __init__(self, goal_pose, Tf = 10, method = 5, ctrl_rate = 150, cmd_type = 1, ctrl_type = 4, K_p = 20.0, vel_limits = vel_limits):

        # Subscribe to current joint state (position, velocity, effort)
        self._sub        = rospy.Subscriber('/arm/joint_states', JointState, self._joint_states_cb)
        
        # Publishers that send command directly to the robot arm
        self._pub_twist  = rospy.Publisher('/arm/joint_group_velocity_controller/command', Float64MultiArray, queue_size=10)        

        # Robot state
        self.arm_joint_names  = []
        self.position         = [0,0,0,0,0,0]
        self.velocity         = [0,0,0,0,0,0]
        self.arm_dof          = len(arm_joint_names)

        # Robot commands        
        self.Vel_msg          = Float64MultiArray()
        self.Vel_msg.data     = [0,0,0,0,0,0]
        self.Pos_msg          = Float64MultiArray()
        self.Pos_msg.data     = [0,0,0,0,0,0]
        self.cmd_type         = cmd_type
        
        # Publishers for debugging        
        self.JSCmd_msg          = JointState()
        self.JSCmd_msg.position = [0,0,0,0,0,0]
        self.JSCmd_msg.velocity = [0,0,0,0,0,0]
        self.JS_msg             = JointState()
        self.JS_msg.position    = [0,0,0,0,0,0]
        self.JS_msg.velocity    = [0,0,0,0,0,0]


        # Control Variables
        self.desired_position   = [0,0,0,0,0,0]
        self.desired_velocity   = [0,0,0,0,0,0]
        self.vel_limits         = vel_limits 

        # Variables for Trajectory Generation (see Chapter 9 of Modern Robotics)
        self.dt     = 1.0/float(ctrl_rate)        
        self.Tf     = Tf
        self.N      = int((self.Tf/self.dt) + 1)        
        self.method = method
        self.iter   = 0

        # Variables for Motion Control (see Chapter 11 of Modern Robotics)
        self.ctrl_type   = ctrl_type
        self.rate        = rospy.Rate(ctrl_rate)
        self.K_p         = K_p
        self.K_i         = pow(K_p,2)/4.0  # To ensure a critically damped error dynamics
        self.theta_e     = np.array([0,0,0,0,0,0])
        self.theta_e_int = np.array([0,0,0,0,0,0])

        # Spin once to get current joint position
        first_msg = rospy.wait_for_message('/arm/joint_states', JointState)
        self._populate_joint_state(first_msg) 
        self.jointstart =  np.array(self.position)
        self.jointend   =  np.array(goal_joint_position)
        
        # For debugging purposes: Visualize init parameters for trajectory
        rospy.loginfo('dt: {}'.format(self.dt))
        rospy.loginfo('Tf: {}'.format(self.Tf))
        rospy.loginfo('N: {}'.format(self.N))
        rospy.loginfo('TrajGen method: {}'.format(self.method))
        rospy.loginfo('Initial joint position: {}'.format(self.jointstart))
        rospy.loginfo('Target  joint position: {}'.format(self.jointend))
        rospy.loginfo('Joint Velocity Limits: {}'.format(self.vel_limits))

        # For debugging purposes: Visualize parameters for Motion Controllers
        rospy.loginfo('Joint Motion Control type: {}'.format(self.ctrl_type))
        if self.ctrl_type>1:
            rospy.loginfo('K_p: {}'.format(self.K_p))
        if self.ctrl_type>2:
            rospy.loginfo('K_i: {}'.format(self.K_i))
        rospy.loginfo('Control command type (1: velocities, 2: positions): {}'.format(self.cmd_type))
        
        # Joint Trajectory Variables
        self.thetamat   = []
        self.dthetamat  = []
        self.ddthetamat = []
        rospy.loginfo('**** Generating trajectory to desired goal target ****')
        self._generate_joint_trajectory()
        rospy.loginfo('**** Ready to run trajectory! ****')


    def _joint_states_cb(self, msg):
        """
            Callback: Populates joint-state variable, computes desired velocities and publishes them
        """
        self._populate_joint_state(msg)   
        # Uncomment if you want to see the current states         
        # rospy.loginfo('Current joint position: {}'.format(self.position))
        # rospy.loginfo('Current joint velocity: {}'.format(self.velocity))    

    def _populate_joint_state(self, msg):
        """ 
            Populates joint state variables. Necessary to resolve for the changes in joint-state message
        """
        self.arm_joint_names = msg.name     
        for i in range(self.arm_dof):
            matched_idx = self.arm_joint_names.index(arm_joint_names[i])
            self.position[i] = msg.position[matched_idx]
            self.velocity[i] = msg.velocity[matched_idx]

    def _generate_joint_trajectory(self):
        """ 
            Computes a straight-line trajectory in joint space
        """
        self.traj = mr.JointTrajectory(self.jointstart, self.jointend, self.Tf, self.N, self.method)

        self.thetamat   = np.array(self.traj).copy()
        self.dthetamat  = np.zeros((self.N, self.arm_dof))
        self.ddthetamat = np.zeros((self.N, self.arm_dof))

        for i in range(np.array(self.traj).shape[0] - 1):
            self.dthetamat[i + 1, :] = (self.thetamat[i + 1, :] - self.thetamat[i, :]) / self.dt
            self.ddthetamat[i + 1, :] \
            = (self.dthetamat[i + 1, :] - self.dthetamat[i, :]) / self.dt

    def _compute_desired_velocities(self):
        """
            Compute desired joint velocities for tracking a reference trajectory
        """
        des_vel = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        
        # Control Type 1: Feed-forward control (Eq. 11.11 from Modern Robotics Book)
        if self.ctrl_type == 1:    
            des_vel = self.dthetamat[self.iter,:]
        else:
            #### Compute Position Error ####
            self.theta_e = self.thetamat[self.iter,:]  - np.array(self.position)                 
            
            #### Compute Integral of Position Error ####
            self.theta_e_int = self.theta_e_int +  self.theta_e*self.dt        
        
            # Control Type 2: Feed-back forward control (P-Controller) (Eq. 11.12 from Modern Robotics Book)            
            if self.ctrl_type == 2:            
                des_vel = self.K_p*self.theta_e

            # Control Type 3: Feed-back forward control (PI-Controller) (Eq. 11.13 from Modern Robotics Book)            
            # ... This controller assumes \dot(theta)_d = \dot(theta) works
            # ... Works perfectly when desired velocity is constant, when not then gains should be tuned 
            if self.ctrl_type == 3:            
                des_vel = self.K_p*self.theta_e + self.K_i*self.theta_e_int

            # Control Type 4: Control Type 1 + Control Type 3  (Eq. 11.15 from Modern Robotics Book)            
            # ... same assumptions as above, but better performance due to ff velocity term
            if self.ctrl_type == 4:
                des_vel = self.dthetamat[self.iter,:] + self.K_p*self.theta_e  +  self.K_i*self.theta_e_int


        # Cap joint velocities with velocity limits 
        for j in range(self.arm_dof):
            if abs(des_vel[j]) > 0.9*self.vel_limits[j]:
                des_vel[j] = des_vel[j]/abs(des_vel[j]) * self.vel_limits[j]

        return des_vel


    def _compute_desired_positions(self):
        """
            Compute desired joint positions from time-dependent reference trajectory
        """
        
        ### Directly from the reference trajectory ###
        # We assume theta_d = theta; i.e. perfect tracking
        des_pos = self.thetamat[self.iter,:]                 

        return des_pos
        
    
    def _publish_desired_velocities(self, des_vel):
        """
            Convert numpy array to Float64MultiArray and publish
        """        
        for j in range(self.arm_dof):
            self.desired_velocity[j] = des_vel[j]

        # Publish command to robot
        self.Vel_msg.data  = self.desired_velocity
        self._pub_vel.publish(self.Vel_msg)

        # Uncomment if you want to see the desired joint velocities sent to the robot        
        # rospy.loginfo('Desired joint velocity: {}'.format(self.desired_velocity))


    def _publish_desired_positions(self, des_pos):
            """
                Convert numpy array to  Float64MultiArray and publish
            """
            for j in range(self.arm_dof):
                 self.desired_position[j] = des_pos[j]

            # Uncomment if you want to see the desired joint positions sent to the robot                         
            rospy.loginfo('Desired joint position: {}'.format(self.desired_position))
            self.Pos_msg.data  = self.desired_position
            self._pub_pos.publish(self.Pos_msg)


    def _publish_jointspace_command(self):
        """
            Publish joint space command for visualization and debugging
        """                
        self.JSCmd_msg.velocity      = self.desired_velocity
        self.JSCmd_msg.position      = self.desired_position
        self.JSCmd_msg.header.stamp  = rospy.Time.now()
        self._pub_JScmd.publish(self.JSCmd_msg)

        # Publish current joint state for visualization
        self.JS_msg.velocity      = self.velocity
        self.JS_msg.position      = self.position
        self.JS_msg.header.stamp  = rospy.Time.now()
        self._pub_JS.publish(self.JS_msg)


    def run(self): 
        """
            The actual control-loop that sends commands (position/velocity) to the robot arm
        """ 
        while not rospy.is_shutdown():


            #### Compute tracking error [Comment if you don't want to see this] ####
            if self.iter > 0:
                pos_track_error = LA.norm(self.thetamat[self.iter-1,:]  - np.array(self.position))                     
                rospy.loginfo('Position tracking error: {}'.format(pos_track_error))

            #### Compute error to target [Comment if you don't want to see this] ####
            error_goal = LA.norm(np.array(self.position) - self.jointend)
            rospy.loginfo('Distance to target: {}'.format(error_goal))

            #### Compute and send control commands to robot arm ####
            if self.cmd_type == 1:
                # Compute desired velocities 
                des_vel = self._compute_desired_velocities()                
                # Publish desired joint-velocities 
                self._publish_desired_velocities(des_vel) 

            if self.cmd_type == 2:
                # Integrate desired velocities to positions
                des_pos = self._compute_desired_positions()                
                # Publish desired joint-positions
                self._publish_desired_positions(des_pos) 

            # Publish desired command for visualization and debugging purposes
            self._publish_jointspace_command()

            # Stop loop/node
            self.iter = self.iter + 1
            if  self.iter > self.N-1:
                rospy.loginfo('**** Reached Tf ****')
                rospy.signal_shutdown('Reached Tf') 
                break

            # Control-loop rate
            self.rate.sleep()
