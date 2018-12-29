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
from intera_core_msgs.msg import EndpointState

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from tf_conversions import posemath
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from tf import transformations

def xyz_to_mat44(pos):
    return transformations.translation_matrix((pos.x, pos.y, pos.z))

def xyzw_to_mat44(ori):
    return transformations.quaternion_matrix((ori.x, ori.y, ori.z, ori.w))

# Callback invoked when EE pose update message is received
# This function will update the pose of object in gripper
def update_ee_pose(msg):
    tf_buffer = tf2_ros.Buffer()    
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    
    obj_pose1 = PoseStamped()
    obj_pose1.header.frame_id = "right_gripper_base"
    obj_pose1.pose.position.x = 0.015
    obj_pose1.pose.position.y = 0.015
    obj_pose1.pose.position.z = 0.14
    
    obj_pose2 = PoseStamped()
    obj_pose2.header.frame_id = "right_gripper_base"
    obj_pose2.pose.position.x = 0.015
    obj_pose2.pose.position.y = -0.015
    obj_pose2.pose.position.z = 0.14
    
    obj_pose3 = PoseStamped()
    obj_pose3.header.frame_id = "right_gripper_base"
    obj_pose3.pose.position.x = -0.015
    obj_pose3.pose.position.y = -0.015
    obj_pose3.pose.position.z = 0.14
        
    
    try:
        tform = tf_buffer.lookup_transform("base",
                                           "right_gripper_base",
                                           rospy.Time(),
                                           rospy.Duration(1.0))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException):
        print("TF Exception, not storing update")
        return
    
    
    trans = (tform.transform.translation.x, tform.transform.translation.y, tform.transform.translation.z)
    rot = (tform.transform.rotation.x, tform.transform.rotation.y, tform.transform.rotation.z, tform.transform.rotation.w)
    mat44 = np.dot(transformations.translation_matrix(trans),
                   transformations.quaternion_matrix(rot))
    
    pose44 = np.dot(xyz_to_mat44(obj_pose1.pose.position),
                    xyzw_to_mat44(obj_pose1.pose.orientation))
    txpose = np.dot(mat44, pose44)
    xyz = tuple(transformations.translation_from_matrix(txpose))[:3]
    x, y, z = xyz
    
    goal_pos_x1 = x
    goal_pos_y1 = y
    goal_pos_z1 = z

    pose44 = np.dot(xyz_to_mat44(obj_pose2.pose.position),
                    xyzw_to_mat44(obj_pose2.pose.orientation))
    txpose = np.dot(mat44, pose44)
    xyz = tuple(transformations.translation_from_matrix(txpose))[:3]
    x, y, z = xyz
    
    goal_pos_x2 = x
    goal_pos_y2 = y
    goal_pos_z2 = z
    
    pose44 = np.dot(xyz_to_mat44(obj_pose3.pose.position),
                    xyzw_to_mat44(obj_pose3.pose.orientation))
    txpose = np.dot(mat44, pose44)
    xyz = tuple(transformations.translation_from_matrix(txpose))[:3]
    x, y, z = xyz
    
    goal_pos_x3 = x
    goal_pos_y3 = y
    goal_pos_z3 = z
    
    print goal_pos_x1, goal_pos_y1, goal_pos_z1
    print goal_pos_x2, goal_pos_y2, goal_pos_z2
    print goal_pos_x3, goal_pos_y3, goal_pos_z3
    print "\n\n"
    
def foo():
    pass

def main():
    rospy.init_node("goal_calculator")
        
    jnt_ee_sub = rospy.Subscriber('/robot/limb/right/endpoint_state',
                                  EndpointState,
                                  update_ee_pose,
                                  queue_size=1)
    while not rospy.is_shutdown():
        foo()
if __name__ == "__main__":
    main()
