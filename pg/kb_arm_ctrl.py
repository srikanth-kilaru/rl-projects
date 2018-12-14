#!/usr/bin/env python

#########################################
# Srikanth Kilaru
# Fall 2018
# MS Robotics, Northwestern University
# Evanston, IL
# srikanthkilaru2018@u.northwestern.edu
##########################################
import sys
import rospy
import roslib
import cv2
import os
import glob
from math import ceil
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import argparse
import actionlib
import imutils
from imutils import paths
import copy
import numpy as np
import logz
import time
import inspect
from multiprocessing import Process
from sklearn import preprocessing
import yaml
import tf
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
from sensor_msgs.msg import JointState
from intera_interface import Limb
from tf import transformations
sys.path.append('/home/srikanth/sawyerws/hrl-kdl/')
sys.path.append('/home/srikanth/sawyerws/hrl-kdl/hrl_geom/src/')
sys.path.append('/home/srikanth/sawyerws/hrl-kdl/pykdl_utils/src')
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics

class KBEnv(object):
    def __init__(self, path):

        rospy.init_node('sawyer_arm_cntrl')
        self.load_inits(path+'/init.yaml')

        self.limb = Limb()
        self.all_jnts = copy.copy(self.limb.joint_names())
        self.limb.set_joint_position_speed(0.2)


    def load_inits(self, path):        
        stream = open(path, "r")
        config = yaml.load(stream)
        stream.close()

        # these indices and names have to be in ascending order
        self.jnt_indices = config['jnt_indices']
        self.cmd_names = config['cmd_names']
        self.init_pos = config['initial-position']


    def chng_jnt_state(self, values, mode, iterations=20):
        # Command Sawyer's joints to angles or velocity
        robot = URDF.from_parameter_server()
        kdl_kin_r = KDLKinematics(robot, 'base',
                                  'right_gripper_r_finger_tip')
        kdl_kin_l = KDLKinematics(robot, 'base',
                                  'right_gripper_l_finger_tip')
        q_jnt_angles = np.zeros(8)
        q_jnt_angles[:7] = copy.copy(values)
        q_jnt_angles[7] = 0.0
        pose_r = kdl_kin_r.forward(q_jnt_angles)
        pose_l = kdl_kin_l.forward(q_jnt_angles)
        pose = 0.5*(pose_l + pose_r)
        x, y, z = transformations.translation_from_matrix(pose)[:3]
        print("goal cartesian: ", x, y, z)
        
        timeout = 30.0
        if mode == 4:
            positions = dict()
            i = 0
            for jnt_name in self.all_jnts:
                positions[jnt_name] = values[i]
                i += 1

            self.limb.move_to_joint_positions(positions, timeout)
        else:
            velocities = dict()
            i = 0
            for name in self.cmd_names:
                velocities[name] = values[i]
                i += 1
            for _ in range(iterations):
                self.limb.set_joint_velocities(velocities)
                time.sleep(0.5)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()
    path = args.path

    env = KBEnv(path)
    
    while not rospy.is_shutdown():
        arr = []
        mode = raw_input("Please enter mode: ")
        mode = int(mode)
        print("Enter comma separated list of joint velocities or positions in ascending order of joint names: ")
        cmds = [float(n) for n in raw_input().split(",")]
        env.chng_jnt_state(cmds, mode)

if __name__ == "__main__":
    main()
