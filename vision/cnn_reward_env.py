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
import cv2 as cv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from intera_interface import Limb


class Env(object):
    def __init__(self, path, train_mode=True):
        self.train_mode = train_mode
        self.obs_lock = threading.Lock()

        # path where the python script for agent and env reside
        self.path = path
        string = "sawyer_ros_cnn"

        rospy.init_node(string + "_environment")
        self.logp = None
        if self.train_mode:
            self.logpath = "CNN_Rew" + '_' + time.strftime("%d-%m-%Y_%H-%M")
            self.logpath = os.path.join(path + '/data', self.logpath)
            if not(os.path.exists(self.logpath)):
                # we should never have to create dir, as agent already done it
                os.makedirs(self.logpath) 
            logfile = os.path.join(self.logpath, "ros_env.log")
            self.logp = open(logfile, "w")

        
        self._load_inits(path)
        self.cur_obs = np.zeros(self.obs_dim)
        self.cur_act = np.zeros(self.act_dim)
        self.cur_reward = None
        self.goal_cntr = 0
        
        self.limb = Limb()
        self.all_jnts = copy.copy(self.limb.joint_names())
        self.limb.set_joint_position_speed(0.2)

        self.rate = rospy.Rate(10)

        self.jnt_st_sub = rospy.Subscriber('/robot/joint_states',
                                           JointState,
                                           self._update_jnt_state,
                                           queue_size=1)
        self.jnt_cm_pub = rospy.Publisher('/robot/limb/right/joint_command',
                                          JointCommand, queue_size=None)    
        self.jnt_ee_sub = rospy.Subscriber('/robot/limb/right/endpoint_state',
                                           EndpointState,
                                           self.update_ee_pose,
                                           queue_size=1)
        self.rgb_sub = rospy.Subscriber("/camera/rgb/image_raw",
                                        Image, rew.rgb_image_subscr)

        

    '''
    All members of the observation vector have to be provided
    '''
    def _set_cur_obs(self, obs):
        #self._print_env_log('Waiting for obs lock')
        self.obs_lock.acquire()
        try:
            #self._print_env_log('Acquired obs lock')
            self.cur_obs = copy.copy(obs)
        finally:
            #self._print_env_log('Released obs lock')
            self.obs_lock.release()

    def _get_cur_obs(self):
        self.obs_lock.acquire()
        try:
            #self._print_env_log('Acquired obs lock')
            obs = copy.copy(self.cur_obs)
        finally:
            #self._print_env_log('Released obs lock')
            self.obs_lock.release()
        return obs

    def close_env_log(self):
        self.logp.close()
        self.logp = None
        
    def _print_env_log(self, string):
        if self.train_mode: 
            if self.logp is None:
                return
            self.logp.write("\n")
            now = rospy.get_rostime()
            t_str = time.strftime("%H-%M")
            t_str += "-" + str(now.secs) + "-" + str(now.nsecs) + ": "
            self.logp.write(t_str + string)
            self.logp.write("\n")
            
    def _load_inits(self, path):
        if self.train_mode:
            # Store versions of the main code required for
            # test and debug after training
            copyfile(path + "/cnn_reward_init.yaml",
                     self.logpath + "/init.yaml")
            copyfile(path + "/cnn_reward_env.py",
                     self.logpath+"/cnn_reward_env.py")

        stream = open(path + "/cnn_reward_init.yaml", "r")
        config = yaml.load(stream)
        stream.close()
        self.distance_thresh = config['distance_thresh']
        # limits for the uniform distribution to
        # sample from when varying initial joint states during training
        # joint position limits have to be in ascending order of joint number
        # jnt_init_limits is a ordered list of [lower_limit, upper_limit] for
        # each joint in motion
        self.jnt_init_limits = config['jnt_init_limits']
        self.cmd_mode = config['cmd_mode']
        if self.cmd_mode == 'VELOCITY_MODE':
            # limits for the joint positions
            self.jnt_pos_limits = config['jnt_pos_limits']
            self.vel_limit = config['vel_limit']
            self.vel_mode = config['vel_mode']
        else:
            self.torque_limit = config['torque_limit']

        '''
        # The following are the names of Sawyer's joints that will move
        # for the shape sorting cube task
        # Any one or more of the following in ascending order -
        # ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4',
        # 'right_j5', 'right_j6']
        # The last joint is the gripper finger joint and stays fixed
        '''
        self.debug_lvl = config['debug_lvl']
        self.cmd_names = config['cmd_names']
        self.init_pos = config['initial-position']
        self.goal_obs_dim = config['goal_obs_dim']
        '''
        In TORQUE_MODE, jnt_obs_dim will be twice the size of the
        number of joints being controlled (2 * self.act_dim), one each for
        position and velocity. 
        The ordering in the observation vector is:
        j0:pos, j1:pos, ..., jN:pos,
        j0:vel, j1:vel ...., jN:vel, # Applicable only in torque mode
        obj_coord:x1, obj_coord:y1, obj_coord:z1, 
        obj_coord:x2, obj_coord:y2, obj_coord:z2,
        obj_coord:x3, obj_coord:y3, obj_coord:z3; 
        goal_coord:x1, goal_coord:y1, goal_coord:z1,
        goal_coord:x2, goal_coord:y2, goal_coord:z2,
        goal_coord:x3, goal_coord:y3, goal_coord:z3
        '''
        self.jnt_obs_dim = config['jnt_obs_dim']
        self.obs_dim = self.goal_obs_dim + self.jnt_obs_dim
        self.act_dim = config['act_dim']
        '''
        These indices have to be in ascending order
        The length of this ascending order list >=1 and <=7 (values 0 to 6)
        The last joint is the gripper finger joint and stays fixed
        '''
        self.jnt_indices = config['jnt_indices']
        # test time goals specified in jnt angle space (for now)
        if not self.train_mode:
            self.test_goal =  config['test_goal']
        self.max_training_goals = config['max_training_goals']
        self.batch_size = config['min_timesteps_per_batch']
        self.goal_classes = config['goal_classes']
                
    # Callback invoked when EE pose update message is received
    def update_ee_pose(self, msg):
        obs = self._get_cur_obs()

        obs[self.jnt_obs_dim]   = msg.pose.position.x
        obs[self.jnt_obs_dim+1] = msg.pose.position.y
        obs[self.jnt_obs_dim+2] = msg.pose.position.z

        '''
        self._print_env_log("EE coordinates: "
                            + str(msg.pose.position.x) +
                            ", " + str(msg.pose.position.y) + ", " +
                            str(msg.pose.position.z))
        '''
        self._set_cur_obs(obs)
        
    def _update_jnt_state(self, msg):
        '''
        Only care for mesgs which have length 9
        There is a length 1 message for the gripper finger joint
        which we dont care about
        '''
        if len(msg.position) != 9:
            return
        
        obs = self._get_cur_obs()
        enum_iter = enumerate(self.jnt_indices, start=0)
        for i, index in enum_iter:
            '''
            Need to add a 1 to message index as it starts from head_pan
            [head_pan, j0, j1, .. torso]
            whereas joint_indices are starting from j0
            '''
            obs[i] = msg.position[index+1]
            if self.cmd_mode == "TORQUE_MODE":
                obs[i + self.jnt_obs_dim/2] = msg.velocity[index+1]

        self._set_cur_obs(obs)
        
    '''
    This function is called from reset only and during both training and testing
    '''
    def _init_jnt_state(self):
        q_jnt_angles = copy.copy(self.init_pos)
        
        # Command Sawyer's joints to pre-set angles and velocity
        # JointCommand.msg mode: TRAJECTORY_MODE
        positions = dict()

        # Build some randomness in starting position
        # between each subsequent iteration
        enum_iter = enumerate(self.jnt_indices, start=0)
        for i, index in enum_iter:
            l_limit = self.jnt_init_limits[i][0]
            u_limit = self.jnt_init_limits[i][1]
            val = np.random.uniform(l_limit, u_limit)
            q_jnt_angles[index] = val

        string = "Initializing joint states to: "
        enum_iter = enumerate(self.all_jnts, start=0)
        for i, jnt_name in enum_iter:
            positions[jnt_name] = q_jnt_angles[i]
            string += str(positions[jnt_name]) + " "
        
        self.limb.move_to_joint_positions(positions, 30)
        self._print_env_log(string)
        # sleep for a bit to ensure that the joints reach commanded positions
        rospy.sleep(3)
        
    def _action_clip(self, action):
        if self.cmd_mode == "TORQUE_MODE":
            return action
            #return np.clip(action, -self.torque_limit, self.torque_limit)
        else:
            return np.clip(action, -self.vel_limit, self.vel_limit)

    def _set_joint_velocities(self, actions):
        if self.vel_mode == "raw":
            velocities = dict()
            enum_iter = enumerate(self.cmd_names, start=0)
            for i, jnt in enum_iter:
                velocities[jnt] = actions[i]
            command_msg = JointCommand()
            command_msg.names = velocities.keys()
            command_msg.velocity = velocities.values()
            command_msg.mode = JointCommand.VELOCITY_MODE
            command_msg.header.stamp = rospy.Time.now()
            self.jnt_cm_pub.publish(command_msg)
        else:
            # Command Sawyer's joints to angles as calculated by velocity*dt
            positions = dict()
            q_jnt_angles = copy.copy(self.init_pos)
            obs_prev = self._get_cur_obs()
            enum_iter = enumerate(self.jnt_indices, start=0)
            for i, index in enum_iter:
                timestep = self.dt + np.random.normal(0, 1)
                val = obs_prev[i] + actions[i] * timestep
                val = np.clip(val, self.jnt_pos_limits[i][0],
                              self.jnt_pos_limits[i][1])
                q_jnt_angles[index] = val

            enum_iter = enumerate(self.all_jnts, start=0)
            for i, jnt_name in enum_iter:
                positions[jnt_name] = q_jnt_angles[i]
                
            self.limb.move_to_joint_positions(positions)
            
    def _set_joint_torques(self, actions):
        torques = dict()
        enum_iter = enumerate(self.all_jnts, start=0)
        for i, jnt_name in enum_iter:
            torques[jnt_name] = 0.0
        enum_iter = enumerate(self.cmd_names, start=0)
        for i, jnt_name in enum_iter:
            torques[jnt_name] = actions[i]
            
        command_msg = JointCommand()
        command_msg.names = torques.keys()
        command_msg.effort = torques.values()
        command_msg.mode = JointCommand.TORQUE_MODE
        command_msg.header.stamp = rospy.Time.now()
        self.jnt_cm_pub.publish(command_msg)
        
    def step(self, action):
        self.cur_act = copy.deepcopy(action)
        
        # called to take a step with the provided action
        # send the action as generated by policy (clip before sending)
        clipped_acts = self._action_clip(action)
        if self.cmd_mode == "TORQUE_MODE":
            self._set_joint_torques(clipped_acts)
        else:
            self._set_joint_velocities(clipped_acts)

        '''
        NOTE: Observations are being constantly updated because
        we are subscribed to the robot state publisher and also
        subscribed to the ee topic which calculates 
        the pose of the goal and the block.
        Sleep for some time, so that the action 
        execution on the robot finishes and we wake up to 
        pick up the latest observation
        '''
        # no sleep necessary if we send velocity integrated positions
        if self.cmd_mode == "TORQUE_MODE" or self.vel_mode == "raw": 
            self.rate.sleep()
        
        obs = self._get_cur_obs()
        self.cur_reward = self._calc_reward()
        return obs, self.cur_reward, self.cur_reward > 0.0

    def _set_test_goal(self):    
        self.goal_class_id = self.test_goal
        
    def _set_random_training_goal(self):
        k = self.goal_cntr
        self.goal_class_id = self.goal_classes[k]
           
    def reset(self):
        # called Initially when the Env is initialized
        # set the initial joint state and send the command
        
        if not self.train_mode:
            self._set_test_goal()
        else:
            if self.goal_cntr == self.max_training_goals - 1:
                self.goal_cntr = 0
            else:
                self.goal_cntr += 1
            self._set_random_training_goal()

        # this call will result in sleeping for 3 seconds
        self._init_jnt_state()
        # send the latest observations
        return self._get_cur_obs()

    def _calc_reward(self):
        
        self._print_env_log(" Current Reward: " + string)        
        return reward
