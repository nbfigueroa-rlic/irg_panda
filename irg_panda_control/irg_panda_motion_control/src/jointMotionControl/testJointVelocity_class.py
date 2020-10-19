import rospy, math
from sensor_msgs.msg import JointState
from std_msgs.msg    import Float64MultiArray
from franka_core_msgs.msg import JointCommand, RobotState
import numpy as np
from copy import deepcopy

# Globally defined variables
PI = math.pi
arm_joint_names = ['panda_joint1','panda_joint2','panda_joint3','panda_joint4','panda_joint5','panda_joint6','panda_joint7']


desired_velocity_pos = [0.75, 0.75, 0.75, 0, 0, 0]  
desired_velocity_neg = [-0.75, -0.75, -0.75, 0, 0, 0]  
limits_pos = [0.90*(PI), 0.5, 1]      
limits_neg = [-0.90*(PI), -0.5, 1]      

class TestJointVelocity(object):
    """
    This class sends velocities to the first three joints of the UR10 robot arm until joint limits are reached.
    """

    def __init__(self):

        # Subscribe to robot joint state
        self._sub_js = rospy.Subscriber('/panda_simulator/custom_franka_state_controller/joint_states', JointState, self._joint_states_cb)

        # Subscribe to robot state (Refer JointState.msg to find all available data. 
        # Note: All msg fields are not populated when using the simulated environment)
        self._sub_rs = rospy.Subscriber('/panda_simulator/custom_franka_state_controller/robot_state', RobotState, self._robot_states_cb)
        
        # Published joint command
        self._pub = rospy.Publisher('/panda_simulator/motion_controller/arm/joint_commands',JointCommand, queue_size = 1, tcp_nodelay = True)

        # Robot state
        self.arm_joint_names  = []
        self.position         = [0,0,0,0,0,0,0]
        self.velocity         = [0,0,0,0,0,0,0]
        self.arm_dof          = len(arm_joint_names)

        # Create JointCommand message to publish commands
        self.pubmsg       = JointCommand()
        self.pubmsg.names = arm_joint_names # names of joints (has to be 7 and in the same order as the command fields (positions, velocities, efforts))
        self.pubmsg.mode  = self.pubmsg.VELOCITY_MODE # Specify control mode (POSITION_MODE, VELOCITY_MODE, IMPEDANCE_MODE (not available in sim), TORQUE_MODE

        # Internal Variables        
        self.go_pos           = [1,1,1]
        self.pos_limit        = [0,0,0]
        self.neg_limit        = [0,0,0]
        self.desired_velocity = [0,0,0,0,0,0,0]
        self.filt             = 0.49999
        self.time             = 0

    def _joint_states_cb(self, msg):
        """
        Callback: Populates joint-state variable, computes desired velocities and publishes them
        """
        self.arm_joint_names = msg.name

        # Necessary to resolve for the changes in joint-state message
        for i in range(self.arm_dof):
            matched_idx = self.arm_joint_names.index(arm_joint_names[i])
            self.position[i] = msg.position[matched_idx]
            self.velocity[i] = msg.velocity[matched_idx]
            
        rospy.loginfo('Current joint position: {}'.format(self.position))
        rospy.loginfo('Current joint velocity: {}'.format(self.velocity))

        # Control loop will be define by joint-states callback
        self._compute_desired_velocities()
        self._pub.publish(self.pubmsg)


    def _robot_states_cb(self, msg):
        if self.time%100 == 0:
            self.time = 1
            rospy.loginfo("============= Current robot state: ============\n" )
            rospy.loginfo("Cartesian vel: \n{}\n".format(msg.O_dP_EE) )
            rospy.loginfo("Gravity compensation torques: \n{}\n".format(msg.gravity) )
            rospy.loginfo("Coriolis: \n{}\n".format(msg.coriolis) )
            rospy.loginfo("Inertia matrix: \n{}\n".format(msg.mass_matrix) )
            rospy.loginfo("Zero Jacobian: \n{}\n".format(msg.O_Jac_EE) )


            rospy.loginfo("\n\n========\n\n")

        self.time+=1

    def _compute_desired_velocities(self):
        """
        Compute desired velocity
        """
        for i in range(3):
            if self.position[i]> limits_pos[i]:
                self.pos_limit[i] = 1
                self.neg_limit[i] = 0
            
            if self.position[i]< limits_neg[i]:
                self.pos_limit[i] = 0
                self.neg_limit[i] = 1

            if self.go_pos[i]==1 and self.pos_limit[i]==1:
                self.go_pos[i]  = 0                           
            elif self.go_pos[i]==0 and self.neg_limit[i]==1:
                self.go_pos[i]  = 1


            if self.go_pos[i]:
                self.desired_velocity[i]   = (1-self.filt)*self.velocity[i] + self.filt*desired_velocity_pos[i]
            else:
                self.desired_velocity[i]   =  (1-self.filt)*self.velocity[i] + self.filt*desired_velocity_neg[i]

        rospy.loginfo('Desired joint velocity: {}'.format(self.desired_velocity))
        self.pubmsg.velocity = self.desired_velocity        

    def run(self): 
        rospy.spin()
