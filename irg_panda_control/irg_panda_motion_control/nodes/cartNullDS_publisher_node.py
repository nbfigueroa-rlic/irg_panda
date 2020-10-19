#! /usr/bin/env python
from __future__ import print_function
import rospy, math, sys
from cartMotionControl.nullDS_class import CartesianMotionControl_NullDS


if __name__ == '__main__':

    rospy.init_node('cartesian_nullDS_publisher')    

	####### Motion Control Variables #######
    ctrl_rate  = 1000 # 150hz

    ####### Initialize Class #######
    cartNullDS = CartesianMotionControl_NullDS(ctrl_rate)

    ####### Run Control #######
    cartNullDS.run()
