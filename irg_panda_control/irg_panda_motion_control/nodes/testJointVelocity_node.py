#! /usr/bin/env python
import rospy
from jointMotionControl.testJointVelocity_class import TestJointVelocity

if __name__ == '__main__':
    
    rospy.init_node('testJointVelocities')
    rospy.wait_for_service('/controller_manager/list_controllers')
    rospy.loginfo("Starting node...")
    rospy.sleep(1)

    testJointVelocity = TestJointVelocity()
    testJointVelocity.run()
