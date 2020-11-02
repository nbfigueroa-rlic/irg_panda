from __future__ import print_function
import rospy, math, time
from sensor_msgs.msg import JointState
from std_msgs.msg    import Float64MultiArray
from geometry_msgs.msg import PointStamped
import numpy as np
from   numpy import linalg as LA
import modern_robotics as mr

# To compute orientation error in quaternion representation
import pyquaternion as pyQuat
Quaternion = pyQuat.Quaternion
import quaternion


from franka_core_msgs.msg import JointCommand, RobotState, EndPointState
import numpy as np

# To visualize target if using JTDS
from visualization_msgs.msg import Marker, MarkerArray

# Globally defined variables
PI = math.pi
arm_joint_names = ['panda_joint1','panda_joint2','panda_joint3','panda_joint4','panda_joint5','panda_joint6','panda_joint7']


# Franka Panda Emika Limits and DH parameters: https://frankaemika.github.io/docs/control_parameters.html
# Limits for Kinematics Class
panda_maxPos   = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
panda_minPos   = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
panda_maxVel   = [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]
panda_maxAcc   = [15, 7.5, 10, 12.5, 15, 20, 20]

# Modified DH Parameters!!! I think I need standard for the for Kinematics Class to work!
panda_DH_a      = [0.0,   0.0,     0.0,     0.0825, -0.0825, 0.0,    0.088]
panda_DH_d      = [0.333, 0.0,     0.316,   0.0,     0.384,  0.0,    0.0]
panda_DH_alpha  = [0.0,   -PI/2.0, PI/2.0,  PI/2.0, -PI/2.0, PI/2.0, PI/2.0 ]
panda_DH_theta0 = [0.0,   0.0,     0.0,     0.0,     0.0,    0.0,    0.0]

# Relative pose between world and arm-base
T_base_rel = np.array([  [1,  0,  0, 0],
                         [0,  1,  0, 0],
                         [0,  0,  1,  0.625],
                          [0, 0, 0, 1]])


T_ee_panda   = np.array([[1, 0, 0, 0], 
                         [0 , 1 , 0, 0],
                         [0, 0, 1, 0.058],
                         [0, 0, 0, 1]])

class JointMotionControl_StateDependent(object):
    """
        This class sends joint velocities to the Franka Panda Robot by following a state-dependent linear Dynamical System in Joint Space 
        in joint space that reaches a joint-position goal (DS_type=1) or task-space goal (DS_type=2))
    """
    def __init__(self, DS_type = 1, A = [0.0] * 7, DS_attractor = [0.0] * 7, ctrl_rate = 1000, cmd_type = 1, epsilon = 0.005, vel_limits = panda_maxVel, null_space = [0.0] * 7):

        # Subscribe to robot joint state
        self._sub_js = rospy.Subscriber('/panda_simulator/custom_franka_state_controller/joint_states', JointState, 
            self._joint_states_cb,
            queue_size=1,tcp_nodelay=True)
        
        # if not using franka_ros_interface, you have to subscribe to the right topics
        # to obtain the current end-effector state and robot jacobian for computing 
        # commands
        self._sub_cart_state = rospy.Subscriber('panda_simulator/custom_franka_state_controller/tip_state', EndPointState, 
            self._on_endpoint_state, 
            queue_size=1,tcp_nodelay=True)

        self._sub_robot_state = rospy.Subscriber('panda_simulator/custom_franka_state_controller/robot_state',RobotState,
            self._on_robot_state,
            queue_size=1,tcp_nodelay=True)
    

        # Published joint command
        self._pub = rospy.Publisher('/panda_simulator/motion_controller/arm/joint_commands',JointCommand, queue_size = 1, tcp_nodelay = True)
        self.DS_type       = DS_type
        if self.DS_type == 2:
            self._pub_target = rospy.Publisher('DS_target', PointStamped)

        # Robot state
        self.arm_joint_names  = []
        self.position         = [0,0,0,0,0,0,0]
        self.velocity         = [0,0,0,0,0,0,0]
        self.arm_dof          = len(arm_joint_names)

        # Create JointCommand message to publish commands
        self.cmd_type      = cmd_type
        self.pubmsg        = JointCommand()
        self.pubmsg.names  = arm_joint_names # names of joints (has to be 7 and in the same order as the command fields (positions, velocities, efforts))
        if self.cmd_type == 1:
            self.pubmsg.mode  = self.pubmsg.VELOCITY_MODE # Specify control mode (POSITION_MODE, VELOCITY_MODE, IMPEDANCE_MODE (not available in sim), TORQUE_MODE
        elif self.cmd_type == 2:
            self.pubmsg.mode  = self.pubmsg.POSITION_MODE

        # Robot Jacobian and EE representations
        self.jacobian           = []
        self.ee_pose            = []
        self.ee_pos             = []
        self.ee_rot             = []
        self.ee_quat            = []
        self.ee_lin_vel         = []
        self.ee_ang_vel         = []
        
        # Control Variables
        self.desired_velocity   = [0,0,0,0,0,0,0]
        self.desired_position   = [0,0,0,0,0,0,0]
        self.vel_limits         = np.array(panda_maxVel)
        self.add_nullspace      = 0

        # Variables for DS-Motion Generator
        self.dt            = 1.0/float(ctrl_rate)                
        self.DS_attractor  = np.array(DS_attractor)        
        self.rate          = rospy.Rate(ctrl_rate)
        self.A             = np.array(A)        
        self.epsilon       = epsilon    
        self.pos_error     = 0
        self.quat_error    = 0     
        self._stop         = 0 
        self.force_limits  = 0

        # Spin once to update robot state
        first_msg = rospy.wait_for_message('/joint_states', JointState)
        self._populate_joint_state(first_msg)         

        # Given size of DS-attractor define FK and Jacobian functions
        # Only for JT-DS type
        if self.DS_type == 2:
            self.DS_pos_att = self.DS_attractor[0:3]
            if len(self.DS_attractor) == 3:
                self.attr_type = 'pos'
            if len(self.DS_attractor) == 7:
                self.attr_type = 'full'    
                self.DS_quat_att = self.DS_attractor[3:7]    

            self.add_nullspace = 0
            self.null_space    = np.array(null_space)    
            self.S_limits      = np.identity(self.arm_dof)            

        # For debugging purposes: Visualize init parameters for trajectory
        rospy.loginfo('DS type (1:joint-space, 2: joint-space task oriented): {}'.format(self.DS_type))
        
        # Specific DS parameters
        if self.DS_type == 1:
            rospy.loginfo('DS attractor: {}'.format(self.DS_attractor))        
        if self.DS_type == 2:
            rospy.loginfo('DS position attractor: {}'.format(self.DS_pos_att))
            rospy.loginfo('Null-Space target: {}'.format(self.null_space))
            if len(self.DS_attractor) == 7:
                rospy.loginfo('DS quaternion attractor: {}'.format(self.DS_quat_att))

        # Generic DS parameters    
        rospy.loginfo('DS system matrix A:\n {}'.format(self.A))
        rospy.loginfo('Stopping epsilon: {}'.format(self.epsilon))

        # For debugging purposes: Visualize parameters for Motion Controllers
        rospy.loginfo('Control command type (1: velocities, 2: positions): {}'.format(self.cmd_type))

    def _joint_states_cb(self, msg):
        """
            Callback: Populates joint-state variable, computes desired velocities and publishes them
        """
        self._populate_joint_state(msg)   
        # Uncomment if you want to see the current states         
        rospy.loginfo('Current joint position: {}'.format(self.position))
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

    def _on_robot_state(self, msg):
        """
            Callback function for updating jacobian and EE velocity from robot state
        """
        # global JACOBIAN, CARTESIAN_VEL
        self.jacobian = np.asarray(msg.O_Jac_EE).reshape(6,7,order = 'F')
        # CARTESIAN_VEL = {'linear': np.asarray([msg.O_dP_EE[0], msg.O_dP_EE[1], msg.O_dP_EE[2]]),
        #                 'angular': np.asarray([msg.O_dP_EE[3], msg.O_dP_EE[4], msg.O_dP_EE[5]]) }
        self.ee_lin_vel =  np.asarray([msg.O_dP_EE[0], msg.O_dP_EE[1], msg.O_dP_EE[2]])
        self.ee_ang_vel =  np.asarray([msg.O_dP_EE[3], msg.O_dP_EE[4], msg.O_dP_EE[5]])

    def _on_endpoint_state(self, msg):
        """
            Callback function to get current end-point state
        """
        # pose message received is a vectorised column major transformation matrix
        cart_pose_trans_mat = np.asarray(msg.O_T_EE).reshape(4,4,order='F')
        # CARTESIAN_POSE = {
        #     'position': cart_pose_trans_mat[:3,3],
        #     'orientation': quaternion.from_rotation_matrix(cart_pose_trans_mat[:3,:3]) }
        
        self.ee_pose = np.dot(np.dot(T_base_rel,cart_pose_trans_mat),T_ee_panda)
        self.ee_pos  = self.ee_pose[0:3,3]

        # self.ee_rot  = np.array([self.ee_pose[0,0:3], self.ee_pose[1,0:3], self.ee_pose[2,0:3]])
        self.ee_rot  = cart_pose_trans_mat[:3,:3]
        # self.ee_quat = quaternion.from_rotation_matrix(self.ee_rot)
        self.ee_quat = Quaternion(matrix=self.ee_rot)

        # rospy.loginfo('\nCurrent ee-pos:\n {}'.format(self.ee_pos))
        # rospy.loginfo('\nCurrent ee-quat:\n {}'.format(self.ee_quat))

        # This might not be necessary
        # if len(self.DS_attractor) == 7:
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

        # This is specific to enforce joint limits!!!
        S  = np.identity(self.arm_dof)   

        # Joint-Space DS Equation
        if self.DS_type == 1:            
            # Compute Joint Error
            joint_error = self.position - self.DS_attractor
            self.pos_error  = LA.norm(joint_error)
            # Compute Desired Velocity
            des_vel = -self.A.dot(joint_error)
            # des_vel = 2.0*des_vel_/LA.norm(des_vel_)
            # des_vel = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        if self.DS_type == 2:
            # Compute Forward Kinematics and Jacobian for JT-DS --> Not necessary if given by 

            # JTDS to track task-space target
            if self.attr_type == 'pos':
                # position only
                task_error = self.ee_pos - self.DS_pos_att    
                self.pos_error  = LA.norm(task_error)

            elif self.attr_type == 'full':
                # position + orientation 
                task_error = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                
                # --- Position error --- #
                task_error[0:3] = self.ee_pos - self.DS_pos_att
                self.pos_error  = LA.norm(task_error[0:3])

                # --- Quaternion error --- #
                quat_attr_    = Quaternion(array=[self.DS_quat_att[3],self.DS_quat_att[0],self.DS_quat_att[1],self.DS_quat_att[2]])

                # Difference between two quaternions
                delta_quat        = self.ee_quat * quat_attr_.conjugate
                delta_log         = Quaternion.log(delta_quat) 
                delta_log_        = np.array([delta_log.elements[1], delta_log.elements[2], delta_log.elements[3]])
                self.quat_error   = LA.norm(delta_log_)
                task_error[3:6]   = 0.5*delta_log_

            J_T          = np.transpose(self.jacobian)            
            joint_error  = J_T.dot(task_error)

            if self.force_limits:
                self._compute_Slimits(-joint_error)

            # Add Limits on Joint Positions
            SA       = self.S_limits.dot(self.A)
            SAS_T    = SA.dot(np.transpose(self.S_limits))

            # Compute Desired Velocity
            if self.pos_error < 0.35 and self.pos_error > self.epsilon*1.1:
                des_vel  = -3*SAS_T.dot(joint_error)       
            else:
                des_vel  = -SAS_T.dot(joint_error)       

            # Add the nullspace velocities to stay in a favoured configuration
            # This only works for 7-DOF robots!!! or for 6-DOF if task-space target is position ONLY
            # This functionality needs to be testeeed
            if self.add_nullspace:            
                JTJ = J_T.dot(self.jacobian)
                rospy.loginfo('JTJ: {}'.format(JTJ))
                P = np.identity(self.arm_dof) - JTJ
                null_error = self.position - self.null_space
                rospy.loginfo('Null-space Error: {}'.format(null_error))
                vel_null = 0.5*P.dot(null_error)
                rospy.loginfo('Null-space Velocities: {}'.format(vel_null))
                des_vel =  des_vel - vel_null


        rospy.loginfo('Raw Desired Velocity: {}'.format(des_vel))

        # # Cap joint velocities with velocity limits        
        for j in range(self.arm_dof):
            if abs(des_vel[j]) > (0.9*self.vel_limits[j]):
                des_vel[j] = (des_vel[j]/abs(des_vel[j])) * 0.90 * self.vel_limits[j]

        rospy.loginfo('Filtered Desired Velocity: {}'.format(des_vel))

        return des_vel

    def _compute_Slimits(self, des_vel_dir):
        """
            Computes S matrix for joint limits (restricted range of motion)
        """
        p = 7

        restr_minPos = 0.99*np.array(panda_minPos)
        restr_maxPos = 0.99*np.array(panda_maxPos)

        # Add range of motion restrictions
        if self.position[1]<-PI/2.0 and des_vel_dir[1] > 0: 
            restr_minPos[1] = -PI*(5.0/4.0)
            rospy.loginfo('MINNN 1111111')            
        if des_vel_dir[1] < 0: 
            restr_maxPos[1] = -PI/8.0
            rospy.loginfo('MAXX 2222222')            

        for j in range(self.arm_dof):
            q_ratio = (self.position[j] - restr_minPos[j])/(restr_maxPos[j] - restr_minPos[j])
            self.S_limits[j,j] = abs(1 - pow(2*q_ratio-1,2*p))
        
        # rospy.loginfo('S limits matrix: {}'.format(self.S_limits))            
    
    def _publish_desired_velocities(self, des_vel):
        """
            Convert numpy array to Float64MultiArray and publish
        """        
        for j in range(self.arm_dof):
            self.desired_velocity[j] = des_vel[j]

        # Publish command to robot
        self.pubmsg.velocity = self.desired_velocity        
        self._pub.publish(self.pubmsg)

        # Uncomment if you want to see the desired joint velocities sent to the robot        
        rospy.loginfo('Desired joint velocity: {}'.format(self.desired_velocity))


    def _publish_desired_positions(self, des_vel):
        """
            Convert numpy array to Float64MultiArray and publish
        """        
        for j in range(self.arm_dof):
            self.desired_position[j] = self.position[j] + des_vel[j]*self.dt

        # Publish command to robot
        self.pubmsg.position = self.desired_position        
        self._pub.publish(self.pubmsg)

        # Uncomment if you want to see the desired joint velocities sent to the robot        
        rospy.loginfo('Desired joint velocity: {}'.format(self.desired_position))

    def _publish_DS_target(self):
           p = PointStamped()
           p.header.frame_id = "/world"
           p.header.stamp = rospy.Time.now()
           
           p.point.x = self.DS_pos_att[0]
           p.point.y = self.DS_pos_att[1]
           p.point.z = self.DS_pos_att[2]

           self._pub_target.publish( p )


    def run(self): 
        """
            The actual control-loop that sends commands (position/velocity) to the robot arm
        """ 

        t0 = time.time()
        reached = 0
        # while not rospy.is_shutdown() and not self._stop:
        while not rospy.is_shutdown() and not reached:
            # Compute desired velocities from DS
            des_vel = self._compute_desired_velocities()              

            #### Send control commands to robot arm ####
            if self.cmd_type == 1:                
                # Publish desired joint-velocities 
                self._publish_desired_velocities(des_vel) 
            elif self.cmd_type == 2:                
                # Integrate desired velocities to positions and publish desired joint-positions
                self._publish_desired_positions(des_vel) 

            #### Compute error to target [Comment if you don't want to see this] ####
            if self.DS_type == 1:
                rospy.loginfo('Distance to target: {}'.format(self.pos_error))
            else: 
                rospy.loginfo('\nDesired ee-pos:\n {}'.format(self.DS_pos_att))
                # rospy.loginfo('\nCurrent ee-pos:\n {}'.format(self.ee_pos))
                rospy.loginfo('Distance to pos-target: {}'.format(self.pos_error))
                rospy.loginfo('Distance to quat-target: {}'.format(self.quat_error))    
                self._publish_DS_target()            
                        
            if  self.pos_error < self.epsilon:
                if self.DS_type == 1:
                    reached = 1
                if self.quat_error < 5*self.epsilon:        
                    reached = 1

            if reached == 1:
                # Send 0 velocities when target reached!                
                if self.cmd_type == 1:                
                    for ii in range(10):           
                        self._publish_desired_velocities(np.array([0, 0, 0, 0, 0, 0, 0]))

                tF = time.time() - t0
                rospy.loginfo('Final Execution time: {}'.format(tF))
                rospy.loginfo('**** Target Reached! ****')
                # rospy.signal_shutdown('Reached target') 
                break        

            # Control-loop rate
            self.rate.sleep()
        return True