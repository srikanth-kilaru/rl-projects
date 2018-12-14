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
import tensorflow as tf
import time
import inspect
from multiprocessing import Process
from sklearn import preprocessing
import yaml
import imp

from intera_motion_interface import (
    MotionTrajectory,
    MotionWaypoint,
    MotionWaypointOptions
)
from intera_motion_msgs.msg import TrajectoryOptions
from geometry_msgs.msg import PoseStamped
import intera_interface
import intera_external_devices
from intera_interface import Limb

def open_jaw_full():
    gripper = intera_interface.Gripper('right_gripper')
    gripper.set_position(gripper.MAX_POSITION)

def gripper_pose(x, y, z):
    limb = Limb()
    traj_options = TrajectoryOptions()
    traj_options.interpolation_type = TrajectoryOptions.CARTESIAN
    traj = MotionTrajectory(trajectory_options = traj_options, limb = limb)
    
    wpt_opts = MotionWaypointOptions()
    waypoint = MotionWaypoint(options = wpt_opts.to_msg(), limb = limb)
    
    joint_names = limb.joint_names()
    endpoint_state = limb.tip_state('right_hand')
    if endpoint_state is None:
        print('Endpoint state not found')
        return False
    pose = endpoint_state.pose
    
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    
    pose.orientation.x = 0
    pose.orientation.y = 0
    pose.orientation.z = 0
    pose.orientation.w = 1
    
    poseStamped = PoseStamped()
    poseStamped.pose = pose
    waypoint.set_cartesian_pose(poseStamped, 'right_hand', [])
    
    traj.append_waypoint(waypoint.to_msg())
    
    result = traj.send_trajectory()
    if result is None:
        print("Trajectory FAILED to send")
        return False
    
    if result.result:
        return True
    else:
        print('Motion controller failed to complete the trajectory %s',
              result.errorId)
        return False

    
def test_pg_policy(file_path, path_len, sim=False):
    env_file = file_path + "/ros_env.py"
    ef = imp.load_source('ros_env', env_file)
    env = ef.Env(file_path, train_mode=False, sim_mode=sim)
        
    with tf.Session() as sess:

        saver = tf.train.import_meta_graph(file_path + "/model.ckpt.meta")
        saver.restore(sess, file_path + "/model.ckpt")
        
        sy_sampled_ac = tf.get_default_graph().get_tensor_by_name("add:0")
        sy_ob_no = tf.get_default_graph().get_tensor_by_name("ob:0")

        ob = env.reset()
        cntr = 0
        while not rospy.is_shutdown():
            ac = sess.run(sy_sampled_ac,
                          feed_dict={sy_ob_no: ob[None]})
            ac = ac[0]
            #print("action = {}".format(ac))
            ob, rew, done = env.step(ac)
            #print("ob = {}".format(ob))

            if done:
                print "Reached the goal"
                #gripper_pose(ob[-6], ob[-5], ob[-4])
                open_jaw_full()
                exit()
            
            if not sim:
                time.sleep(0.2)
            cntr += 1
            if cntr > path_len:
                print "Could not reach the goal"
                exit()
            
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--sim', action='store_true')
    args = parser.parse_args()
    file_path = args.path
    sim = args.sim
    stream = open(file_path + "/init.yaml", "r")
        
    config = yaml.load(stream)
    stream.close()    
    seed = config['seed']
    path_len = config['max_path_length']
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
    test_pg_policy(file_path, path_len, sim)
    
if __name__ == "__main__":
    main()
