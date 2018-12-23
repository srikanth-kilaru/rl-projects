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
'''
from intera_motion_interface import (
    MotionTrajectory,
    MotionWaypoint,
    MotionWaypointOptions
)
'''
from intera_interface import Limb
from tf import transformations
sys.path.append('/home/srikanth/sawyerws/hrl-kdl/')
sys.path.append('/home/srikanth/sawyerws/hrl-kdl/hrl_geom/src/')
sys.path.append('/home/srikanth/sawyerws/hrl-kdl/pykdl_utils/src')
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics

def xyz_to_mat44(pos):
    return transformations.translation_matrix((pos.x, pos.y, pos.z))

def xyzw_to_mat44(ori):
    return transformations.quaternion_matrix((ori.x, ori.y, ori.z, ori.w))

class Env(object):
    def __init__(self, path, train_mode=True, sim_mode=False):
        self.train_mode = train_mode
        self.sim_mode = sim_mode
        self.obs_lock = threading.Lock()

        # path where the python script for agent and env reside
        self.path = path
        if not sim_mode:
            string = "sawyer_ros"
        else:
            string = "sim"
        rospy.init_node(string + "_environment")
        self.logp = None
        if self.train_mode:
            self.logpath = "PG" + '_' + time.strftime("%d-%m-%Y_%H-%M")
            self.logpath = os.path.join(path + '/data', self.logpath)
            if not(os.path.exists(self.logpath)):
                # we should never have to create dir, as agent already done it
                os.makedirs(self.logpath) 
            logfile = os.path.join(self.logpath, "ros_env.log")
            self.logp = open(logfile, "w")

        
        # Goal does not move and is specified by three points (for pose reasons)
        self.goal_pos_x1 = None
        self.goal_pos_y1 = None
        self.goal_pos_z1 = None
        self.goal_pos_x2 = None
        self.goal_pos_y2 = None
        self.goal_pos_z2 = None
        self.goal_pos_x3 = None
        self.goal_pos_y3 = None
        self.goal_pos_z3 = None
        
        # dt - time step used for simulation
        self.dt = 0.1
        self._load_inits(path)
        self.cur_obs = np.zeros(self.obs_dim)
        self.cur_act = np.zeros(self.act_dim)
        self.cur_reward = None
        self.goal_cntr = 0
        
        self.limb = Limb()
        self.all_jnts = copy.copy(self.limb.joint_names())
        self.limb.set_joint_position_speed(0.2)

        self.robot = URDF.from_parameter_server()
        self.kdl_kin_r = KDLKinematics(self.robot, 'base',
                                       'right_gripper_r_finger_tip')
        self.kdl_kin_l = KDLKinematics(self.robot, 'base',
                                       'right_gripper_l_finger_tip')
        
        if not self.sim_mode:
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

        '''
        Three points on the object in "right_gripper_base" frame.
        Obect is static in gripper frame. Needs to be calculated only once
        during init time
        '''
        self.obj_pose1 = PoseStamped()
        self.obj_pose1.header.frame_id = "right_gripper_base"
        self.obj_pose1.pose.position.x = self.obj_x1
        self.obj_pose1.pose.position.y = self.obj_y1
        self.obj_pose1.pose.position.z = self.obj_z1

        self.obj_pose2 = PoseStamped()
        self.obj_pose2.header.frame_id = "right_gripper_base"
        self.obj_pose2.pose.position.x = self.obj_x2
        self.obj_pose2.pose.position.y = self.obj_y2
        self.obj_pose2.pose.position.z = self.obj_z2

        self.obj_pose3 = PoseStamped()
        self.obj_pose3.header.frame_id = "right_gripper_base"
        self.obj_pose3.pose.position.x = self.obj_x3
        self.obj_pose3.pose.position.y = self.obj_y3
        self.obj_pose3.pose.position.z = self.obj_z3
        
        self.tf_buffer = tf2_ros.Buffer()    
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)


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
            copyfile(path + "/init.yaml",
                     self.logpath + "/init.yaml")
            copyfile(path + "/ros_env.py",
                     self.logpath+"/ros_env.py")
            copyfile(path + "/pg_agent.py",
                     self.logpath + "/pg_agent.py")
            copyfile(path + "/pg_test.py",
                     self.logpath + "/pg_test.py")

        stream = open(path + "/init.yaml", "r")
        config = yaml.load(stream)
        stream.close()
        self.distance_thresh = config['distance_thresh']
        # limits for the uniform distribution to
        # sample from when varying initial joint states during training
        # joint position limits have to be in ascending order of joint number
        # jnt_init_limits is a ordered list of [lower_limit, upper_limit] for
        # each joint in motion
        self.jnt_init_limits = config['jnt_init_limits']
        # limits for the joint positions
        self.jnt_pos_limits = config['jnt_pos_limits']
        self.cmd_mode = config['cmd_mode']
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
        self.vel_limit = config['vel_limit']
        self.torque_limit = config['torque_limit']
        self.vel_mode = config['vel_mode']
        # test time goals specified in jnt angle space (for now)
        self.test_goal =  config['test_goal']
        self.max_training_goals = config['max_training_goals']
        self.batch_size = config['min_timesteps_per_batch']
        self.goal_XYZs = config['goal_XYZs']
        '''
        The following 9 object coordinates are in "right_gripper_base" frame
        They are relayed by the camera observer at init time.
        They can be alternatively read from the init.yaml file too
        '''
        self.obj_x1 = config['obj_x1']
        self.obj_y1 = config['obj_y1']
        self.obj_z1 = config['obj_z1']
        self.obj_x2 = config['obj_x2']
        self.obj_y2 = config['obj_y2']
        self.obj_z2 = config['obj_z2']
        self.obj_x3 = config['obj_x3']
        self.obj_y3 = config['obj_y3']`
        self.obj_z3 = config['obj_z3']
        
    # Callback invoked when EE pose update message is received
    # This function will update the pose of object in gripper
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

        obs = self._get_cur_obs()

        trans = (tform.transform.translation.x, tform.transform.translation.y, tform.transform.translation.z)
        rot = (tform.transform.rotation.x, tform.transform.rotation.y, tform.transform.rotation.z, tform.transform.rotation.w)
        mat44 = np.dot(transformations.translation_matrix(trans),
                       transformations.quaternion_matrix(rot))

        pose44 = np.dot(xyz_to_mat44(self.obj_pose1.pose.position),
                        xyzw_to_mat44(self.obj_pose1.pose.orientation))
        txpose = np.dot(mat44, pose44)
        xyz = tuple(transformations.translation_from_matrix(txpose))[:3]
        x, y, z = xyz

        obs[self.jnt_obs_dim]   = x 
        obs[self.jnt_obs_dim+1] = y 
        obs[self.jnt_obs_dim+2] = z 

        pose44 = np.dot(xyz_to_mat44(self.obj_pose2.pose.position),
                        xyzw_to_mat44(self.obj_pose2.pose.orientation))
        txpose = np.dot(mat44, pose44)
        xyz = tuple(transformations.translation_from_matrix(txpose))[:3]
        x, y, z = xyz

        obs[self.jnt_obs_dim+3] = x 
        obs[self.jnt_obs_dim+4] = y 
        obs[self.jnt_obs_dim+5] = z 

        pose44 = np.dot(xyz_to_mat44(self.obj_pose3.pose.position),
                        xyzw_to_mat44(self.obj_pose3.pose.orientation))
        txpose = np.dot(mat44, pose44)
        xyz = tuple(transformations.translation_from_matrix(txpose))[:3]
        x, y, z = xyz

        obs[self.jnt_obs_dim+6] = x 
        obs[self.jnt_obs_dim+7] = y 
        obs[self.jnt_obs_dim+8] = z 

        self._set_cur_obs(obs)
        
    def _update_obj_coords(self, q_jnt_angles):
        # forward kinematics (returns homogeneous 4x4 numpy.mat)
        pose_r = self.kdl_kin_r.forward(q_jnt_angles)
        pose_l = self.kdl_kin_l.forward(q_jnt_angles)
        #object face center is right in between the two tip frames
        pose = 0.5*(pose_l + pose_r)
        x, y, z = transformations.translation_from_matrix(pose)[:3]
        return x, y, z
        
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
    for Real and sim environments
    '''
    def _init_jnt_state(self):
        q_jnt_angles = copy.copy(self.init_pos)
        
        if self.sim_mode:
            '''
            enum_iter = enumerate(self.jnt_indices, start=0)
            obs = self._get_cur_obs()
            for i, index in enum_iter:
                # Build some randomness between each subsequent iteration
                l_limit = self.jnt_init_limits[i][0]
                u_limit = self.jnt_init_limits[i][1]
                val = np.random.uniform(l_limit, u_limit)
                # Update the cur obs of the joints in play 
                obs[i] = val
                q_jnt_angles[index] = val

            x, y, z = self._update_obj_coords(q_jnt_angles)
            obs[self.jnt_obs_dim]   = x
            obs[self.jnt_obs_dim+1] = y
            obs[self.jnt_obs_dim+2] = z
            self._set_cur_obs(obs)
            string = str(self.init_pos)
            self._print_env_log("Initializing joint states to " + string)
            return # its a sim, no need to send ROS messages
            '''
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
        
        if self.sim_mode:
            '''
            # NOTE: sim_mode only supports velocity and not torque control mode
            # Usage of sim_mode in torque control mode will lead to garbage

            # Integrate action also updates current obs
            self._sim_integrate_action(self.cur_act)
            # q_jnt_angles has j0-j6 and also finger position. Total size 8
            q_jnt_angles = copy.copy(self.init_pos)
            obs = self._get_cur_obs()
            enum_iter = enumerate(self.jnt_indices, start=0)
            for i, index in enum_iter:
                q_jnt_angles[index] = obs[i]

            x, y, z = self._update_obj_coords(q_jnt_angles)
            obs[self.jnt_obs_dim]   = x
            obs[self.jnt_obs_dim+1] = y
            obs[self.jnt_obs_dim+2] = z
            self._set_cur_obs(obs)
            '''
        else:
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
        diff = self._get_diff(obs)
        done = self._is_done(diff)
        self.cur_reward = self._calc_reward(diff, done)
        return obs, self.cur_reward, done

    def _set_cartesian_test_goal(self):    
        self.goal_pos_x1 = self.test_goal[0][0]
        self.goal_pos_y1 = self.test_goal[0][1]
        self.goal_pos_z1 = self.test_goal[0][2]
        self.goal_pos_x1 = self.test_goal[1][0]
        self.goal_pos_y1 = self.test_goal[1][1]
        self.goal_pos_z1 = self.test_goal[1][2]
        self.goal_pos_x1 = self.test_goal[2][0]
        self.goal_pos_y1 = self.test_goal[2][1]
        self.goal_pos_z1 = self.test_goal[2][2]

    def _set_random_training_goal(self):
        k = self.goal_cntr
        self.goal_pos_x1 = self.goal_XYZs[k][0][0]
        self.goal_pos_y1 = self.goal_XYZs[k][0][1]
        self.goal_pos_z1 = self.goal_XYZs[k][0][2]
        self.goal_pos_x2 = self.goal_XYZs[k][1][0]
        self.goal_pos_y2 = self.goal_XYZs[k][1][1]
        self.goal_pos_z2 = self.goal_XYZs[k][1][2]
        self.goal_pos_x3 = self.goal_XYZs[k][2][0]
        self.goal_pos_y3 = self.goal_XYZs[k][2][1]
        self.goal_pos_z3 = self.goal_XYZs[k][2][2]
           
    def _sim_integrate_action(self, action):
        obs = self.dt*action
        obs += np.random.normal(0, 1) # add some noise
        cur_obs = self._get_cur_obs()
        cur_obs[:self.jnt_obs_dim] += obs
        self._set_cur_obs(cur_obs)
        
    def reset(self, t_steps=0):
        # called Initially when the Env is initialized
        # set the initial joint state and send the command
        
        if not self.train_mode:
            self._set_cartesian_test_goal()
        else:
            if self.goal_cntr == self.max_training_goals - 1:
                self.goal_cntr = 0
            else:
                self.goal_cntr += 1
            self._set_random_training_goal()

        cur_obs = self._get_cur_obs()
        # Update cur_obs with the new goal
        od = self.jnt_obs_dim
        cur_obs[od+9]  = self.goal_pos_x1
        cur_obs[od+10] = self.goal_pos_y1
        cur_obs[od+11] = self.goal_pos_z1
        cur_obs[od+12] = self.goal_pos_x2
        cur_obs[od+13] = self.goal_pos_y2
        cur_obs[od+14] = self.goal_pos_z2
        cur_obs[od+15] = self.goal_pos_x3
        cur_obs[od+16] = self.goal_pos_y3
        cur_obs[od+17] = self.goal_pos_z3
        
        self._set_cur_obs(cur_obs)

        # this call will result in sleeping for 3 seconds for non-sim env
        self._init_jnt_state()
        # send the latest observations
        return self._get_cur_obs()

    def _get_diff (self, obs):
        od = self.jnt_obs_dim

        diff = [abs(obs[od] - obs[od+9]),
                abs(obs[od+1] - obs[od+10]),
                abs(obs[od+2] - obs[od+11]),
                abs(obs[od+3] - obs[od+12]),
                abs(obs[od+4] - obs[od+13]),
                abs(obs[od+5] - obs[od+14]),
                abs(obs[od+6] - obs[od+15]),
                abs(obs[od+7] - obs[od+16]),
                abs(obs[od+8] - obs[od+17])]

        return diff
    
    def _is_done(self, diff):
        # all elements of d are positive values
        done = all(d <= self.distance_thresh for d in diff)
        if done:
            self._print_env_log(" Reached the goal!!!! ")
        return done

    def _calc_reward(self, diff, done):
        l2 = np.linalg.norm(np.array(diff))
        l2sq = l2**2
        alpha = 1e-6
        w_l2 = -1e-3
        w_u = -1e-2
        w_log = -1.0
        reward = 0.0
        dist_cost = l2sq
        precision_cost = np.log(l2sq + alpha)
        cntrl_cost = np.square(self.cur_act).sum()
        
        reward += w_l2 * dist_cost + w_log * precision_cost + w_u * cntrl_cost
        
        string = "l2sq: {}, log of l2sq: {} contrl_cost: {} reward: {}".format(dist_cost,
                                                                               precision_cost,
                                                                               cntrl_cost,
                                                                               reward)
        self._print_env_log(" Current Reward: " + string)        
        return reward
