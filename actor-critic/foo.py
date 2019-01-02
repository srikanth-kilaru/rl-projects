import sys
import rospy
import roslib
import os
import glob
import argparse
import actionlib
import imutils
from imutils import paths
import copy
import numpy as np
import time
import yaml
from shutil import copyfile
import threading
import tf2_ros
import intera_interface
import intera_external_devices
from intera_interface import CHECK_VERSION
from intera_motion_msgs.msg import TrajectoryOptions
from intera_core_msgs.msg import EndpointState
from intera_core_msgs.msg import JointCommand
from geometry_msgs.msg import PoseStamped
from tf_conversions import posemath
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from intera_interface import Limb
from tf import transformations

def xyz_to_mat44(pos):
        return transformations.translation_matrix((pos.x, pos.y, pos.z))

def xyzw_to_mat44(ori):
        return transformations.quaternion_matrix((ori.x, ori.y, ori.z, ori.w))

class Env(object):
        def __init__(self):
                rospy.init_node("test_environment")
                self.rate = rospy.Rate(10)
                self.tf_buffer = tf2_ros.Buffer()    
                self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
                self.jnt_ee_sub = rospy.Subscriber('/robot/limb/right/endpoint_state',
                                                   EndpointState,
                                                   self.update_ee_pose,
                                                   queue_size=1)
                self.obj_pose1 = PoseStamped()
                self.obj_pose1.header.frame_id = "right_gripper_base"
                self.obj_pose1.pose.position.x = 0.015
                self.obj_pose1.pose.position.y = -0.015
                self.obj_pose1.pose.position.z = 0.14

        
        def update_ee_pose(self, msg):
                try:
                        tform = self.tf_buffer.lookup_transform("base",
                                                                "right_gripper_base",
                                                                rospy.Time(),
                                                                rospy.Duration(1.0))
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                        tf2_ros.ExtrapolationException):
                        self._print_env_log("TF Exception, not storing update")
                        return

                trans = (tform.transform.translation.x, tform.transform.translation.y, tform.transform.translation.z)
                rot = (tform.transform.rotation.x, tform.transform.rotation.y, tform.transform.rotation.z, tform.transform.rotation.w)
                mat44 = np.dot(transformations.translation_matrix(trans),
                               transformations.quaternion_matrix(rot))
                
                pose44 = np.dot(xyz_to_mat44(self.obj_pose1.pose.position),
                                xyzw_to_mat44(self.obj_pose1.pose.orientation))
                txpose = np.dot(mat44, pose44)
                xyz = tuple(transformations.translation_from_matrix(txpose))[:3]
                x, y, z = xyz
                
                print x, y, z
                
def main():
        Env()
        while not rospy.is_shutdown():
                pass
        
if __name__ == "__main__":
    main()
    
