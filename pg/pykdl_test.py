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
sys.path.append('/home/srikanth/sawyerws/hrl-kdl/')
sys.path.append('/home/srikanth/sawyerws/hrl-kdl/hrl_geom/src/')
sys.path.append('/home/srikanth/sawyerws/hrl-kdl/pykdl_utils/src')
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from tf import transformations

def xyz_to_mat44(x, y, z):
    return transformations.translation_matrix((x, y, z))

def main():
    robot = URDF.from_parameter_server()
    print robot.get_root()
    tree = kdl_tree_from_urdf_model(robot)
    print tree.getNrOfSegments()
    chain = tree.getChain(robot.get_root(), 'right_gripper_r_finger_tip')
    print chain.getNrOfJoints()

    init_pos = [1.605525390625, -3.06816015625, 1.708900390625, 0.0580078125, 1.364841796875, 0.12111677875185967, 0.721302320838, 0.036830651785714284]
    jnt_indices = [3,5]
    q_jnt_angles = copy.copy(init_pos)
    l_pos_limit = -2.0
    u_pos_limit = 2.0

    kdl_kin_r = KDLKinematics(robot, robot.get_root(),
                              'right_gripper_r_finger_tip')
    kdl_kin_l = KDLKinematics(robot, robot.get_root(),
                              'right_gripper_l_finger_tip')
    print kdl_kin_l.get_joint_names()
    print kdl_kin_l.get_joint_limits()
    #kdl_kin_r.extract_joint_state(js)
    

    '''
    for index in jnt_indices:
        # Do a -1 on the index because we are indexing into
        # a list that does not include the sawyer head pan
        # index includes the head pan
        q_jnt_angles[index] = np.random.uniform(l_pos_limit,
                                                  u_pos_limit)
    '''
    print q_jnt_angles
    # forward kinematics (returns homogeneous 4x4 numpy matrix)
    pose_r = kdl_kin_r.forward(q_jnt_angles)
    pose_l = kdl_kin_l.forward(q_jnt_angles)
    print "pose_l: ", pose_l
    print "pose_r: ", pose_r
    pose = 0.5*(pose_l + pose_r)
    x, y, z = transformations.translation_from_matrix(pose)[:3]

    print x, y, z

    q_ik_l = kdl_kin_l.inverse(pose_l, q_jnt_angles) # inverse kinematics
    if q_ik_l is not None:
        pose_sol = kdl_kin_l.forward(q_ik_l) # should equal pose
        print 'q_ik_l:', q_ik_l
        print 'pose_sol:', pose_sol

    q_ik_r = kdl_kin_r.inverse(pose_r, q_jnt_angles) # inverse kinematics
    if q_ik_r is not None:
        pose_sol = kdl_kin_r.forward(q_ik_r) # should equal pose
        print 'q_ik_r:', q_ik_r
        print 'pose_sol:', pose_sol

if __name__ == "__main__":
    main()
