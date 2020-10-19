from __future__ import print_function
import rospy, math, time
from sensor_msgs.msg import JointState
from std_msgs.msg    import Float64MultiArray
import numpy as np
from   numpy import linalg as LA
import modern_robotics as mr

# To get Jacobian and FW kinematics (much faster.. 0.5ms for FK and 0.6ms for Jacobian)
from generic_kinematics_model.kinematics import Kinematics

# To compute orientation error in quaternion representation
import pyquaternion as pyQuat
Quaternion = pyQuat.Quaternion

######## Globally defined variables ########
PI = math.pi

arm_joint_names = [
    'joint1', 'joint2', 'joint3',
    'joint4', 'joint5', 'joint6'
]

# Limits for Kinematics Class
ur10_maxPos   = [PI, PI, PI, PI, PI, PI]
ur10_minPos   = [-PI, -PI, -PI, -PI, -PI, -PI]

# MELFA ASSISTA LIMITS [from Spec Sheet]
ur10_maxVel   = [2.16 , 2.16 , 2.16 , 5.18, 6.12, 6.28]

# MELFA ASSISTA LIMITS [tested on the robot -- greater gives errors]
ur10_maxVel   = [2.16 * 0.65, 2.16 * 0.65, 2.16 * 0.65, 5.18* 0.55, 6.12* 0.55, 6.28 * 0.55]


# with 0.55 it gives the power supply error (brake).. 

# 250 mm/s
# DH Parameters for Kinematics Class
ur10_DH_a      = [0.0 ,-0.612 ,-0.5723 , 0.0 , 0.0 , 0.0]
ur10_DH_d      = [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922]
ur10_DH_alpha  = [PI/2.0, 0.0, 0.0, PI/2.0, -PI/2.0, 0.0 ]
ur10_DH_theta0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Relative pose between Robbie-base (world) and arm-base
T_base_rel = np.array([  [-0.0002037,  1.0000000,  0.0000000, 0.33000],
               [-1.0000000, -0.0002037,  0.0000000, 0.00000],
               [0.0000000,  0.0000000,  1.0000000,  0.48600],
               [0.0000000, 0.0000000, 0.0000000, 1.0000000]])

# Necessary for UR robots due to errors in URDF file!
T_base_UR = np.array([[-1.0,  0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0,  0.0],
                    [0.0,   0.0, 1.0,  0.0],                      
                    [0.0 ,  0.0,  0.0, 1.0]])
T_ee_UR   = np.array([[0, -1, 0, 0], 
                    [0, 0, -1, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]])

class JointMotionControl_StateDependent(object):
    """
        This class sends joint velocities/positions to the UR10 by following a time-dependent a straight-line trajectory 
        in joint space that reaches a joint-position goal with different joint velocity control methods
    """
    def __init__(self, DS_type = 1, A = [0.0] * 6, DS_attractor = [0.0] * 6, ctrl_rate = 150, cmd_type = 1, epsilon = 0.005, vel_limits = ur10_maxVel, null_space = [0.0] * 6):

        # Subscribe to current joint state (position, velocity, effort)
        self._sub        = rospy.Subscriber('/joint_states', JointState, self._joint_states_cb)

        # Publishers that send velocity command to a filter (CDDynamics) before sending to robot
        self._pub_vel    = rospy.Publisher('/cobot/desired_joint_velocity', Float64MultiArray, queue_size=10)        

        # Robot Joint States
        self.arm_joint_names    = []
        self.position           = [0,0,0,0,0,0]
        self.velocity           = [0,0,0,0,0,0]
        self.arm_dof            = len(arm_joint_names)        
        
        # Robot Jacobian and EE representations
        self.jacobian           = []
        self.ee_pose            = []
        self.ee_pos             = []
        self.ee_rot             = []
        self.ee_quat            = []

        # Robot commands        
        self.Vel_msg          = Float64MultiArray()
        self.Vel_msg.data     = [0,0,0,0,0,0]
        self.cmd_type         = cmd_type
        

        # Control Variables
        self.desired_velocity   = [0,0,0,0,0,0]
        self.vel_limits         = np.array(ur10_maxVel)
        self.add_nullspace      = 0

        # Variables for DS-Motion Generator
        self.dt            = 1.0/float(ctrl_rate)        
        self.DS_type       = DS_type
        self.DS_attractor  = np.array(DS_attractor)        
        self.rate          = rospy.Rate(ctrl_rate)
        self.A             = np.array(A)        
        self.epsilon       = epsilon    
        self.pos_error     = 0
        self.quat_error    = 0     
        self._stop         = 0 
        self.force_limits  = 1
        self.good_config   = 0

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

            # Initialize Arm Kinematics-Chain Class for FK and Jacobian Computation
            self.arm_kinematics = Kinematics(self.arm_dof)
            for i in range(self.arm_dof):
                self.arm_kinematics.setDH(i, ur10_DH_a[i], ur10_DH_d[i], ur10_DH_alpha[i], ur10_DH_theta0[i], 1, ur10_minPos[i], ur10_maxPos[i], ur10_maxVel[i])
            self.T_base = np.dot(T_base_rel, T_base_UR)
            self.arm_kinematics.setT0(self.T_base)
            self.arm_kinematics.setTF(T_ee_UR)
            self.arm_kinematics.readyForKinematics()
            self._forwardKinematics()
            self._computeJacobian()
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
            # des_vel = 1.5*des_vel_/LA.norm(des_vel_)
            # des_vel = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        if self.DS_type == 2:
            # Compute Forward Kinematics and Jacobian for JT-DS
            self._forwardKinematics()        
            self._computeJacobian()

            # JTDS to track task-space target
            if self.attr_type == 'pos':
                # position only
                task_error = self.ee_pos - self.DS_pos_att    

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
                task_error[3:6]   = delta_log_

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

        # Cap joint velocities with velocity limits        
        # for j in range(self.arm_dof):
        #     if abs(des_vel[j]) > self.vel_limits[j]:
        #         des_vel[j] = des_vel[j]/abs(des_vel[j]) * self.vel_limits[j]

        max_vel = 0.75
        for j in range(self.arm_dof):
            if abs(des_vel[j]) > max_vel:
                des_vel[j] = des_vel[j]/abs(des_vel[j]) * max_vel

        rospy.loginfo('Filtered Desired Velocity: {}'.format(des_vel))

        return des_vel

    def _computeJacobian(self):
        """
            Computes spatial/geometric Jacobian matrix of current joint position
        """
        # Compute Forward Kinematics from current joint state  
        if self.attr_type == 'pos':
            self.jacobian = self.arm_kinematics.getJacobianPos() 
        elif self.attr_type == 'full':
            self.jacobian = self.arm_kinematics.getJacobian()        

    def _forwardKinematics(self):
        """
            Computes forward kinematics of current joint position
        """
        # Convert to desired format for Kinematics() class   
        np_query_joint_pos = np.zeros((self.arm_dof, 1))
        for j in range(self.arm_dof):
            np_query_joint_pos[j,0] = self.position[j]
        self.arm_kinematics.setJoints(np_query_joint_pos, 1)
        
        # Compute Forward Kinematics from current joint state  
        self.ee_pose = self.arm_kinematics.getEndTMatrix()        

        # rospy.loginfo('\nCurrent ee-pose:\n {}'.format(self.ee_pose))
        self.ee_pos  = self.ee_pose[0:3,3]
        self.ee_rot  = np.array([self.ee_pose[0,0:3], self.ee_pose[1,0:3], self.ee_pose[2,0:3]])
        self.ee_quat = Quaternion(matrix=self.ee_rot)

        # Adjust quaternion signs for antipodal issues
        if self.DS_type == 2:
            switch_sign = 0
            if self.DS_quat_att[3] > 0:
                if self.ee_quat.elements[0] < 0:
                    switch_sign = 1

            if self.DS_quat_att[3] < 0:
                if self.ee_quat.elements[0] > 0:
                    switch_sign = 1

            if switch_sign == 1:
                for i in range(4):
                    self.ee_quat[i] = -self.ee_quat[i]

    def _compute_Slimits(self, des_vel_dir):
        """
            Computes S matrix for joint limits (restricted range of motion)
        """
        p = 6

        restr_minPos = 0.99*np.array(ur10_minPos)
        restr_maxPos = 0.99*np.array(ur10_maxPos)

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

    def _moveto_working_range(self):
        """
            If robot is out of the desired working range of motion
        """
    
        rospy.loginfo('Moving robot to good init config...')  
        shoulder_limit = -PI/4
        elbow_limit    = PI/2
        wrist_limit    = -2.303    

        while self.position[1] > shoulder_limit*0.9 and self.position[2] < elbow_limit*0.9:
            des_vel = np.array([0.0]*6)
            des_vel[1] = -5*(self.position[1] - shoulder_limit)
            des_vel[2] = -5*(self.position[2] - elbow_limit)
            des_vel[3] = -5*(self.position[3] - wrist_limit)

            if self.cmd_type == 1:                
                # Publish desired joint-velocities 
                self._publish_desired_velocities(des_vel) 

            if self.cmd_type == 2:
                # Integrate desired velocities to positions
                des_pos = self._compute_desired_positions(des_vel)                
                # Publish desired joint-positions
                self._publish_desired_positions(des_pos) 

    
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
        rospy.loginfo('Desired joint velocity: {}'.format(self.desired_velocity))


    def run(self): 
        """
            The actual control-loop that sends commands (position/velocity) to the robot arm
        """ 
       
        # Check if robot is out of working range and move it there
        if self.DS_type == 2:
            self._moveto_working_range()        
            rospy.loginfo('***** Ready to start DS motion generation! *****') 
        

        t0 = time.time()
        while not rospy.is_shutdown() and not self._stop:
            # Compute desired velocities from DS
            des_vel = self._compute_desired_velocities()              

            #### Send control commands to robot arm ####
            # if self.cmd_type == 1:                
            # Publish desired joint-velocities 
            self._publish_desired_velocities(des_vel) 

            # if self.cmd_type == 2:
            #     # Integrate desired velocities to positions
            #     des_pos = self._compute_desired_positions(des_vel)                
            #     # Publish desired joint-positions
            #     self._publish_desired_positions(des_pos) 

            # Publish desired command for visualization and debugging purposes
            # self._publish_jointspace_command()

            #### Compute error to target [Comment if you don't want to see this] ####
            if self.DS_type == 1:
                rospy.loginfo('Distance to target: {}'.format(self.pos_error))
            else: 
                rospy.loginfo('Distance to pos-target: {}'.format(self.pos_error))
                rospy.loginfo('Distance to quat-target: {}'.format(self.quat_error))                
            
            reached = 0
            if  self.pos_error < self.epsilon:
                if self.DS_type == 1:
                    reached = 1
                if self.quat_error < 5*self.epsilon:        
                    reached = 1

            if reached == 1:
                # Send 0 velocities when target reached!                
                if self.cmd_type == 1:                
                    for ii in range(10):           
                        self._publish_desired_velocities(np.array([0, 0, 0, 0, 0, 0 ]))

                tF = time.time() - t0
                rospy.loginfo('Final Execution time: {}'.format(tF))
                rospy.loginfo('**** Target Reached! ****')
                rospy.signal_shutdown('Reached target') 
                break        

            # Control-loop rate
            self.rate.sleep()
