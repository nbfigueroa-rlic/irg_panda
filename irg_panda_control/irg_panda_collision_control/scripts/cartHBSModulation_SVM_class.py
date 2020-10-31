from __future__ import print_function
import rospy, math, time
from sensor_msgs.msg import JointState
from std_msgs.msg    import Float64MultiArray
from geometry_msgs.msg import Twist, Wrench, Vector3, PointStamped, WrenchStamped
import numpy as np
from   numpy import linalg as LA
# import modern_robotics as mr

# To compute orientation error in quaternion representation
import pyquaternion as pyQuat
Quaternion = pyQuat.Quaternion
# import quaternion

# For franka control
from franka_core_msgs.msg import JointCommand, RobotState, EndPointState
import numpy as np

# For modulation!
import sys
sys.path.append("./dynamical_system_modulation_svm/")
import learn_gamma_fn
from modulation_utils import *
import modulation_svm

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

class CartesianMotionControl_DSModulation(object):
    """
        This class sends desired twists (linear and angular velocity) to the 
    """
    def __init__(self, DS_type = 1, A_p = [0.0] * 3, A_o = [0.0] * 3, DS_attractor = [0.0] * 7, ctrl_rate = 1000, epsilon = 0.005, ctrl_orient = 1, learned_gamma = []):

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
    
        # ---Publishes twist command to filter node
        self._pub_twist    = rospy.Publisher('/panda_simulator/desired_twist', Twist, queue_size=10)        
        self._pub_target   = rospy.Publisher('DS_target', PointStamped, queue_size=1)
        self._pub_ee_pos   = rospy.Publisher('curr_ee_pos', PointStamped, queue_size=1)
        self._pub_wrench   = rospy.Publisher('/panda_simulator/desired_wrench', WrenchStamped, queue_size=10)

        # Robot state
        self.arm_joint_names  = []
        self.position         = [0,0,0,0,0,0,0]
        self.velocity         = [0,0,0,0,0,0,0]
        self.arm_dof          = len(arm_joint_names)

        # Create JointCommand message to publish commands
        self._pubmsg        = JointCommand()
        self._pubmsg.names  = arm_joint_names # names of joints (has to be 7 and in the same order as the command fields (positions, velocities, efforts))
        self._pubmsg.mode   = self._pubmsg.TORQUE_MODE
        # Also create a publisher to publish joint commands
        self._joint_command_publisher = rospy.Publisher(
            'panda_simulator/motion_controller/arm/joint_commands',
            JointCommand,
            tcp_nodelay=True,
            queue_size=1)

        self.control_orient = ctrl_orient

        # Robot Jacobian and EE representations
        self.jacobian           = []
        self.ee_pose            = []
        self.ee_pos             = []
        self.ee_rot             = []
        self.ee_quat            = []
        self.ee_lin_vel         = []
        self.ee_ang_vel         = []
    
        # Robot commands in task-space       
        self.twist_msg           = Twist()
        self.twist_msg.linear.x  = 0
        self.twist_msg.linear.y  = 0
        self.twist_msg.linear.z  = 0
        self.twist_msg.angular.x = 0
        self.twist_msg.angular.y = 0
        self.twist_msg.angular.z = 0     

        # Might remove this, doesn't seem to work like I expect it to..
        self.wrench_msg                  = WrenchStamped()
        self.wrench_msg.wrench           = Wrench()
        self.wrench_msg.wrench.force     = Vector3()
        self.wrench_msg.wrench.torque    = Vector3()
        # self.wrench_msg.header.frame_id  = '/panda_link7'
        # self.wrench_msg.header.frame_id  = '/world_on_ee'
        self.wrench_msg.header.stamp     = rospy.Time.now()

        # Control Variables
        self.desired_position   = [0,0,0,0,0,0]
        self.desired_velocity   = [0,0,0,0,0,0]
        # self.max_lin_vel        = 1.70
        # self.max_ang_vel        = 2.5

        self.max_lin_vel        = 1.0
        self.max_ang_vel        = 1.0

        # Variables for DS-Motion Generator
        self.dt            = 1.0/float(ctrl_rate)        
        self.DS_type       = DS_type
        self.DS_attractor  = np.array(DS_attractor)        
        self.rate          = rospy.Rate(ctrl_rate)
        self.epsilon       = epsilon    
        self._stop         = 0

        # Linear DS     
        self.A_p               = np.array(A_p)
        self.pos_error         = 0 
        self.scale_DS          = 1
        self.learned_gamma     = learned_gamma
        self.classifier        = learned_gamma['classifier']
        self.max_dist          = learned_gamma['max_dist']
        self.reference_points  = learned_gamma['reference_points']

        # Quaternion DS
        self.A_o           = np.array(A_o)                
        self.quat_error    = 0

        # Spin once to update robot state
        first_msg      = rospy.wait_for_message('/panda_simulator/custom_franka_state_controller/joint_states', JointState)
        self._populate_joint_state(first_msg)         

        self.DS_pos_att = self.DS_attractor[0:3]
        if len(self.DS_attractor) == 3:
            self.attr_type = 'pos'
        if len(self.DS_attractor) == 7:
            self.attr_type = 'full'    
            self.DS_quat_att = self.DS_attractor[3:7]    

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


    def _compute_desired_velocities(self):
        """
            Compute desired task-space velocities from state-dependent DS
        """        

        # --- Linear Controller with Modulated DS! --- #
        pos_delta = self.ee_pos - self.DS_pos_att    
        self.pos_error  = LA.norm(pos_delta)        

        # Compute Desired Linear Velocity
        # orig_ds = -self.A_p.dot(pos_delta)
        orig_ds = -pos_delta
        rospy.loginfo("x_dot:{}".format(orig_ds))
        gamma_val  = learn_gamma_fn.get_gamma(np.array(self.ee_pos), self.classifier, self.max_dist, self.reference_points, dimension=3)
        rospy.loginfo("gamma(x):{}".format(gamma_val))
        normal_vec = learn_gamma_fn.get_normal_direction(np.array(self.ee_pos), self.classifier, self.reference_points, self.max_dist, dimension=3)        
        rospy.loginfo("gamma_grad(x):{}".format(normal_vec))

        if gamma_val < 1:
            rospy.loginfo('ROBOT IS COLLIDED REGION.. DO SOMETHING, CHARLIE!!!')

        # if np.dot(orig_ds, normal_vec) > 0:
        #     x_dot_mod  = orig_ds
        # else:
        x_dot_mod  = modulation_svm.modulation_singleGamma_HBS_multiRef(query_pt=np.array(self.ee_pos), orig_ds=np.array(orig_ds), gamma_query=gamma_val,
                            normal_vec_query=normal_vec.reshape(3), obstacle_reference_points= self.reference_points, repulsive_gammaMargin=0.01)
    
        rospy.loginfo("x_dot_mod:{}".format(x_dot_mod))
        lin_vel =  self.scale_DS*x_dot_mod

        # if self.ee_pos[2] > 1.0 and gamma_val < 1:
            # lin_vel[2] = lin_vel[2] + 0.05 

        rospy.loginfo('Desired linear velocity: {}'.format(lin_vel))     

        # --- Compute desired attractor aligned to reference --- #
        desired_rotation = np.array((3,3))
        # if np.linalg.norm(normal_vec) == 0:
        #     R_z = np.array([0,0,-1])    
        # else:
            # R_z = - normal_vec.reshape(3)/np.linalg.norm(normal_vec.reshape(3))
        
        if self.DS_attractor[1] < 0:
            sign_y = -1
        else:
            sign_y = 1 

        R_y = sign_y * orig_ds/np.linalg.norm(orig_ds)     
        rospy.loginfo('R_y: {}'.format(R_y))

        if self.ee_pos[2] > 1: 
            R_z = np.array([0,0,-1])
        else:
            R_z = np.array([0,0,1])            
        rospy.loginfo('R_y: {}'.format(R_y))        

        # Make it orthogonal to n
        R_y -= R_y.dot(R_z) * R_z / np.linalg.norm(R_z)**2
        # normalize it
        R_y /= np.linalg.norm(R_y)
        R_x = np.cross(R_y,R_z)
        desired_rotation = np.hstack([R_x,R_y,R_z])
        rospy.loginfo('R_mod: {}'.format(desired_rotation))        
        quat_attr_ = Quaternion(matrix=self.ee_rot)

        # --- Quaternion error --- #
        # quat_attr_    = Quaternion(array=[self.DS_quat_att[3],self.DS_quat_att[0],self.DS_quat_att[1],self.DS_quat_att[2]])

        # Difference between two quaternions
        delta_quat        = self.ee_quat * quat_attr_.conjugate
        delta_log         = Quaternion.log(delta_quat) 
        delta_log_np      = np.array([delta_log.elements[1], delta_log.elements[2], delta_log.elements[3]])
        self.quat_error   = LA.norm(delta_log_np)

        # --- Compute Desired Angular Velocity --- #
        ang_vel           = -self.A_o.dot(delta_log_np)

        # --- Compute Desired Angular Velocity in ee-reference frame --- #
        RT             = self.ee_rot.transpose()
        # Difference desired quaternion from desired angular velocity
        # q_des  = quat_mul(quat_exp(0.5 * quat_deriv(omega * self.dt )), q_curr);
        delta_quat_des = Quaternion(array=[0, ang_vel[0]*self.dt, ang_vel[1]*self.dt, ang_vel[2]*self.dt])
        q_des          = Quaternion.exp(0.5*delta_quat_des) * self.ee_quat
        RTR_des        = RT.dot(q_des.rotation_matrix)

        #scale angular velocity
        w = 1 - math.tanh(gamma_val)
        rospy.loginfo('w: {}'.format(w))        
        ang_vel_rot    = w*RTR_des.dot(ang_vel)
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


        # Scale velocities if too low
        if LA.norm(lin_vel) < 0.25:
                lin_vel = lin_vel/LA.norm(lin_vel + 1e-5) * 0.25

        rospy.loginfo('||x_dot||: {}'.format(LA.norm(lin_vel)))
        rospy.loginfo('||omega||: {}'.format(LA.norm(ang_vel)))

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


    def _publish_DS_target(self):
           p = PointStamped()
           p.header.frame_id = "/world"
           p.header.stamp = rospy.Time.now()
           
           p.point.x = self.DS_pos_att[0]
           p.point.y = self.DS_pos_att[1]
           p.point.z = self.DS_pos_att[2]

           self._pub_target.publish( p )


    def _publish_EE_position(self):
           p = PointStamped()
           p.header.frame_id = "/world"
           p.header.stamp = rospy.Time.now()
           
           p.point.x = self.ee_pos[0]
           p.point.y = self.ee_pos[1]
           p.point.z = self.ee_pos[2]

           self._pub_ee_pos.publish( p )


    def _compute_desired_wrench(self, lin_vel, ang_vel):
            """
            Computes desired task-space force using PD law
            """   
            # Truncate velocities if too high!
            if LA.norm(lin_vel) > self.max_lin_vel:
                    lin_vel = lin_vel/LA.norm(lin_vel) * self.max_lin_vel

            if LA.norm(ang_vel) > self.max_ang_vel:
                    ang_vel = ang_vel/LA.norm(ang_vel) * self.max_ang_vel


            # Task-space controller parameters
            # K_pos = 50.
            # K_ori = 25.
            K_pos = 500.
            K_ori = 100.
            # damping gains
            D_pos = 20.
            D_ori = 5.
            # D_pos = math.sqrt(4*K_pos)
            # D_ori = math.sqrt(4*K_ori)

            # Desired robot end-effector state
            goal_pos     =  self.ee_pos + lin_vel*self.dt
            rospy.loginfo('Desired Position: {}'.format(goal_pos))            

            # Deltas for PD law
            delta_pos = (goal_pos - self.ee_pos).reshape([3,1])

            # This should be the quaternion difference
            delta_ori = (ang_vel).reshape([3,1]) 
            # delta_ori = self.delta_quat_des.reshape([3,1])
               
            delta_lin_vel = (self.ee_lin_vel - lin_vel).reshape([3,1])
            delta_ang_vel = (self.ee_ang_vel - ang_vel).reshape([3,1])
               
            rospy.loginfo('Position Error: {}'.format(delta_pos))
            rospy.loginfo('Orientation Error: {}'.format(delta_ori))

            # Desired task-space force using PD law
            # F_ee = np.vstack([K_pos*(delta_pos), K_ori*(delta_ori)]) - \
            # np.vstack([D_pos*(delta_lin_vel), D_ori*(self.ee_ang_vel.reshape([3,1]))])

            # if self.control_orient:
            #     F_ee = np.vstack([K_pos*(delta_pos), K_ori*(delta_ori)]) - \
            #     np.vstack([D_pos*(delta_lin_vel), D_ori*(self.ee_ang_vel.reshape([3,1]))])
            # else:     
            F_ee = np.vstack([K_pos*(delta_pos), np.array([0,0,0]).reshape([3,1])]) - \
            np.vstack([D_pos*(delta_lin_vel), np.array([0,0,0]).reshape([3,1])])


            rospy.loginfo('Desired Wrench: {}'.format(F_ee))
            return F_ee


    def _publish_desired_wrench(self, lin_vel, ang_vel):

            # lin_vel_rot = self.ee_rot.dot(np.array(lin_vel))

            self.wrench_msg.wrench.force.x = lin_vel[0]
            self.wrench_msg.wrench.force.y = lin_vel[1]
            self.wrench_msg.wrench.force.z = lin_vel[2]

            self.wrench_msg.wrench.torque.x = ang_vel[0]
            self.wrench_msg.wrench.torque.y = ang_vel[1]
            self.wrench_msg.wrench.torque.z = ang_vel[2]

            self._pub_wrench.publish(self.wrench_msg)        

    def _publish_torque_commands(self, F_ee):            
            # joint torques to be commanded
            J = self.jacobian
            tau = np.dot(J.T,F_ee)

            # publish joint commands
            self._pubmsg.effort = tau.flatten()
            self._joint_command_publisher.publish(self._pubmsg)        

    def run(self): 
        """
            The actual control-loop that sends commands (position/velocity) to the robot arm
        """     
        t0 = time.time()
        reached = 0
        while not rospy.is_shutdown() and not self._stop:

            # Compute desired velocities from DS
            lin_vel, ang_vel = self._compute_desired_velocities()              

            # Publish desired twist
            self._publish_desired_twist(lin_vel, ang_vel) 
            self._publish_DS_target()
            self._publish_EE_position()
            # self._publish_desired_wrench(lin_vel, ang_vel)

            # Compute desired velocities from DS
            F_ee = self._compute_desired_wrench(lin_vel, ang_vel)                       
            self._publish_torque_commands(F_ee)               

            #### Compute error to target [Comment if you don't want to see this] ####
            rospy.loginfo('Distance to pos-target: {}'.format(self.pos_error))
            rospy.loginfo('Distance to quat-target: {}'.format(self.quat_error))                
            
            
            if self.control_orient:
                if(self.pos_error < self.epsilon) and (self.quat_error < 2*self.epsilon):
                    reached = 1
            else:
                if(self.pos_error < self.epsilon):
                    reached = 1

            if reached == 1:
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