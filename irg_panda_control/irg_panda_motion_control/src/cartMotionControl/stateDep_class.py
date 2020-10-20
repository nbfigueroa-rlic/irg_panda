from __future__ import print_function
import rospy, math, time
from sensor_msgs.msg import JointState
from franka_core_msgs.msg import JointCommand, RobotState
from geometry_msgs.msg import Pose, Twist
import numpy as np
from   numpy import linalg as LA
import modern_robotics as mr
from copy import deepcopy

# To compute orientation error in quaternion representation
import pyquaternion as pyQuat
Quaternion = pyQuat.Quaternion

######## Globally defined variables ########
PI = math.pi
names = ['panda_joint1','panda_joint2','panda_joint3','panda_joint4','panda_joint5','panda_joint6','panda_joint7']

class CartesianMotionControl_StateDependent(object):
    """
        This class sends desired twists (linear and angular velocity) to the 
    """
    def __init__(self, DS_type = 1, A_p = [0.0] * 3, A_o = [0.0] * 3, DS_attractor = [0.0] * 7, ctrl_rate = 150, epsilon = 0.005):

        # Subscribe to current joint state (position, velocity, effort)
        self._sub       = rospy.Subscriber('/joint_states', JointState, self._joint_states_cb)
        self._sub_pose  = rospy.Subscriber('/cobot/ee_pose', Pose, self._ee_pose_cb)
        
        # Publishes the current EE pose (from FK of joint-states)
        # self._pub_pose    = rospy.Publisher('/UR10arm/ee_pose', Pose, queue_size=10)                
        
        # --- Publishes the twist command directly to low-level velocity controller of the robot arm
        self._pub_twist   = rospy.Publisher('/arm/twist_controller/command_twist', Twist, queue_size=10)        
        
        # ---Publishes twist command to filter node
        # self._pub_twist   = rospy.Publisher('/UR10arm/desired_twist', Twist, queue_size=10)        
        

        # Robot Joint States
        self.arm_joint_names    = []
        self.position           = [0,0,0,0,0,0]
        self.velocity           = [0,0,0,0,0,0]
        self.arm_dof            = len(arm_joint_names)        
        
        # Robot Jacobian and EE representations
        # self.ee_pose            = []
        self.ee_pos             = [0,0,0]
        # self.ee_rot             = []
        self.ee_quat            = [0,0,0,0]
        # self.ee_vel             = []
    

        # Robot commands in task-space       
        self.twist_msg           = Twist()
        self.twist_msg.linear.x  = 0
        self.twist_msg.linear.y  = 0
        self.twist_msg.linear.z  = 0
        self.twist_msg.angular.x = 0
        self.twist_msg.angular.y = 0
        self.twist_msg.angular.z = 0        

        # Control Variables
        self.desired_position   = [0,0,0,0,0,0]
        self.desired_velocity   = [0,0,0,0,0,0]
        # self.max_lin_vel        = 0.40
        # self.max_ang_vel        = 0.35

        self.max_lin_vel        = 0.200
        self.max_ang_vel        = 0.100

        # Variables for DS-Motion Generator
        self.dt            = 1.0/float(ctrl_rate)        
        self.DS_type       = DS_type
        self.DS_attractor  = np.array(DS_attractor)        
        self.rate          = rospy.Rate(ctrl_rate)
        self.A_p           = np.array(A_p)        
        self.A_o           = np.array(A_o)        
        self.epsilon       = epsilon    
        self.pos_error     = 0
        self.quat_error    = 0     
        self._stop         = 0 

        # Spin once to update robot state
        first_msg      = rospy.wait_for_message('/joint_states', JointState)
        self._populate_joint_state(first_msg)         

        first_msg_pose = rospy.wait_for_message('/cobot/ee_pose', JointState)
        self._populate_ee_pose(first_msg_pose) 

        self.DS_pos_att = self.DS_attractor[0:3]
        if len(self.DS_attractor) == 3:
            self.attr_type = 'pos'
        if len(self.DS_attractor) == 7:
            self.attr_type = 'full'    
            self.DS_quat_att = self.DS_attractor[3:7]    

        # Initialize Arm Kinematics-Chain Class for FK and Jacobian Computation
        # self.arm_kinematics = Kinematics(self.arm_dof)
        # for i in range(self.arm_dof):
        #     self.arm_kinematics.setDH(i, ur10_DH_a[i], ur10_DH_d[i], ur10_DH_alpha[i], ur10_DH_theta0[i], 1, ur10_minPos[i], ur10_maxPos[i], ur10_maxVel[i])
        # self.T_base = np.dot(T_base_rel, T_base_UR)
        # self.arm_kinematics.setT0(self.T_base)
        # self.arm_kinematics.setTF(T_ee_UR)
        # self.arm_kinematics.readyForKinematics()
        # self._forwardKinematics()

        # For debugging purposes: Visualize init parameters for trajectory
        rospy.loginfo('DS type (1:linear, 2: non-linear): {}'.format(self.DS_type))
        
        # Specific DS parameters
        rospy.loginfo('DS position attractor: {}'.format(self.DS_pos_att))
        rospy.loginfo('DS quaternion attractor: {}'.format(self.DS_quat_att))

        # Generic DS parameters    
        rospy.loginfo('DS system matrix A_p:\n {}'.format(self.A_p))
        rospy.loginfo('DS system matrix A_o:\n {}'.format(self.A_o))
        rospy.loginfo('Stopping epsilon: {}'.format(self.epsilon))


    def _joint_states_cb(self, msg):
        """ 
            Call back for joint-state subscriber, populate variables and 
        """    
        # Populate Joint State variables
        self._populate_joint_state(msg)
      

    def _populate_joint_state(self, msg):        
        """ 
            Populates joint state variables. Necessary to resolve for the changes in joint-state message
        """
        self.arm_joint_names = msg.name     
        for i in range(self.arm_dof):
            matched_idx = self.arm_joint_names.index(arm_joint_names[i])
            self.position[i] = msg.position[matched_idx]
            self.velocity[i] = msg.velocity[matched_idx]


    def _ee_pose_cb(self, msg):
        """ 
            Call back for ee-pose subscriber, populate variables and 
        """    
        self._populate_ee_pose(msg)


    def _populate_ee_pose(self, msg):        
        """ 
            Populates joint state variables. Necessary to resolve for the changes in joint-state message
        """        
        # Populate pose message
        self.ee_pos[0]  = msg.position.x 
        self.ee_pos[1]  = msg.position.y 
        self.ee_pos[2]  = msg.position.z 
        self.ee_quat[1] = msg.orientation.x
        self.ee_quat[2] = msg.orientation.y 
        self.ee_quat[3] = msg.orientation.z       
        self.ee_quat[0] = msg.orientation.w      
        self.ee_rot  = Quaternion(self.ee_quat).rotation_matrix 

        # ee_rot = Quaternion(self.ee_quat).rotation_matrix 
        # self.ee_rot  = np.array([ee_rot[0,0:3], ee_rot[1,0:3], ee_rot[2,0:3]])
    

    # def _forwardKinematics(self):
    #     """
    #         Computes forward kinematics of current joint position
    #     """
    #     # Convert to desired format for Kinematics() class   
    #     np_query_joint_pos = np.zeros((self.arm_dof, 1))
    #     for j in range(self.arm_dof):
    #         np_query_joint_pos[j,0] = self.position[j]
    #     self.arm_kinematics.setJoints(np_query_joint_pos, 1)
        
    #     # Compute Forward Kinematics from current joint state  
    #     self.ee_pose = self.arm_kinematics.getEndTMatrix()        

    #     # rospy.loginfo('\nCurrent ee-pose:\n {}'.format(self.ee_pose))
    #     self.ee_pos  = self.ee_pose[0:3,3]
    #     self.ee_rot  = np.array([self.ee_pose[0,0:3], self.ee_pose[1,0:3], self.ee_pose[2,0:3]])
    #     self.ee_quat = Quaternion(matrix=self.ee_rot)

    #     # Adjust quaternion signs for antipodal issues
    #     if self.DS_type == 2:
    #         switch_sign = 0
    #         if self.DS_quat_att[3] > 0:
    #             if self.ee_quat.elements[0] < 0:
    #                 switch_sign = 1

    #         if self.DS_quat_att[3] < 0:
    #             if self.ee_quat.elements[0] > 0:
    #                 switch_sign = 1

    #         if switch_sign == 1:
    #             for i in range(4):
    #                 self.ee_quat[i] = -self.ee_quat[i]        
    
    def _compute_desired_velocities(self):
        """
            Compute desired joint velocities from state-dependent DS
        """        
        
        # --- Position error --- #
        pos_delta = self.ee_pos - self.DS_pos_att    
        self.pos_error  = LA.norm(pos_delta)        

        # Compute Desired Linear Velocity
        lin_vel = -self.A_p.dot(pos_delta)

        # --- Quaternion error --- #
        quat_attr_    = Quaternion(array=[self.DS_quat_att[3],self.DS_quat_att[0],self.DS_quat_att[1],self.DS_quat_att[2]])

        # Difference between two quaternions
        delta_quat        = self.ee_quat * quat_attr_.conjugate
        delta_log         = Quaternion.log(delta_quat) 
        delta_log_np      = np.array([delta_log.elements[1], delta_log.elements[2], delta_log.elements[3]])
        self.quat_error   = LA.norm(delta_log_np)

        # --- Compute Desired Angular Velocity --- #
        ang_vel           = -self.A_o.dot(delta_log_np)
        # rospy.loginfo('omega: {}'.format(ang_vel))        

        # --- Compute Desired Angular Velocity in ee-reference frame --- #
        RT             = self.ee_rot.transpose()
        # Difference desired quaternion from desired angular velocity
        # q_des  = quat_mul(quat_exp(0.5 * quat_deriv(omega * self.dt )), q_curr);
        delta_quat_des = Quaternion(array=[0, ang_vel[0]*self.dt, ang_vel[1]*self.dt, ang_vel[2]*self.dt])
        q_des          = Quaternion.exp(0.5*delta_quat_des) * self.ee_quat
        RTR_des        = RT.dot(q_des.rotation_matrix)
        ang_vel_rot    = RTR_des.dot(ang_vel)
        rospy.loginfo('omega: {}'.format(ang_vel_rot))        

        # Comments: there seems to be no difference when using velocity control.. but maybe 
        # it is necessary for torque-controlled robots... need to check with the kuka simulation or panda?

        return lin_vel, ang_vel_rot
   
    def _publish_desired_twist(self, lin_vel, ang_vel):
        """
            Convert numpy arrays to Twist-msg and publish
        """       
        # Truncate velocities if too high!
        if LA.norm(lin_vel) > self.max_lin_vel:
                lin_vel = lin_vel/LA.norm(lin_vel) * self.max_lin_vel

        if LA.norm(ang_vel) > self.max_ang_vel:
                ang_vel = ang_vel/LA.norm(ang_vel) * self.max_ang_vel

        rospy.loginfo('||x_dot||: {}'.format(LA.norm(lin_vel)))
        rospy.loginfo('||omega||: {}'.format(LA.norm(ang_vel)))

        # Populate twist message
        self.twist_msg.linear.x  = lin_vel[0]
        self.twist_msg.linear.y  = lin_vel[1]
        self.twist_msg.linear.z  = lin_vel[2]
        self.twist_msg.angular.x = ang_vel[0]
        self.twist_msg.angular.y = ang_vel[1]
        self.twist_msg.angular.z = ang_vel[2]
        # self.twist_msg.angular.x = 0
        # self.twist_msg.angular.y = 0
        # self.twist_msg.angular.z = 0


        # Publish command to low-level controller
        self._pub_twist.publish(self.twist_msg)

        # Uncomment if you want to see the desired joint velocities sent to the robot        
        # rospy.loginfo('Desired twist: {}'.format(self.twist_msg))


    def run(self): 
        """
            The actual control-loop that sends commands (position/velocity) to the robot arm
        """     
        t0 = time.time()
        reached = 0
        while not rospy.is_shutdown() and not self._stop:
        
            # Compute Forward Kinematics to get EE-pose
            # self._forwardKinematics()    

            # Publish current EE-pose
            # self._publish_ee_pose()  

            # Compute desired velocities from DS
            lin_vel, ang_vel = self._compute_desired_velocities()              
            
            # Publish desired twist
            self._publish_desired_twist(lin_vel, ang_vel) 

            #### Compute error to target [Comment if you don't want to see this] ####
            rospy.loginfo('Distance to pos-target: {}'.format(self.pos_error))
            rospy.loginfo('Distance to quat-target: {}'.format(self.quat_error))                
            
            
            if  (self.pos_error < self.epsilon) and (self.quat_error < self.epsilon):
                reached = 1
                tF = time.time() - t0     
                # Send 0 velocities when target reached!                
                for ii in range(10):           
                    self._publish_desired_twist(np.array([0, 0, 0]), np.array([0, 0, 0]))

                rospy.loginfo('Final Execution time: {}'.format(tF))
                rospy.loginfo('**** Target Reached! ****')
                rospy.signal_shutdown('Reached target') 
                break        

            # Control-loop rate
            self.rate.sleep()