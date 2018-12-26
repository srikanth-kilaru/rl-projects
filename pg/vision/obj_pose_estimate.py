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
import numpy as np
import cv2
import os
import glob
from math import ceil
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
import argparse
import actionlib
import imutils
from imutils import paths
from datetime import datetime
from matplotlib import pyplot as plt
from scipy import stats
import message_filters
from geometry_msgs.msg import PoseStamped
from tf_conversions import posemath
from geometry_msgs.msg import Point
import calibrate as cb
import math
import tf2_ros
from tf import transformations
from intera_core_msgs.msg import EndpointState
from intera_interface import Limb
from numpy import cos
from numpy import sin
from numpy import pi
import copy
import pyzbar.pyzbar as pyzbar
import pcl
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def nothing(x):
    pass

def xyz_to_mat44(pos):
    return transformations.translation_matrix((pos.x, pos.y, pos.z))

def xyzw_to_mat44(ori):
    return transformations.quaternion_matrix((ori.x, ori.y, ori.z, ori.w))

def transform_camera2robot(X,Y,Z):
    obj_loc_kin = np.array([[X], [Y], [Z], [1]])

    rotZ = np.array([[cos(pi), -sin(pi), 0, 0], [sin(pi), cos(pi), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    rotX = np.array([[1, 0, 0, 0], [0, cos(pi/2), -sin(pi/2), 0], [0, sin(pi/2), cos(pi/2), 0], [0, 0, 0, 1]])
    R = rotX.dot(rotZ)
    
    #do the transformation
    kinect_to_sawyer = R.dot(obj_loc_kin)
    
    kinect_loc = np.array([[0.0], [0.0], [0.6604], [1]])    
    sawyer_to_obj = kinect_to_sawyer + kinect_loc
    return sawyer_to_obj[0][0], sawyer_to_obj[1][0], sawyer_to_obj[2][0]

def getXYZ(xp, yp, zc, fx, fy, cx, cy):
    xn = (xp - cx) / fx
    yn = (yp - cy) / fy
    xc = xn * zc
    yc = yn * zc
    return(xc, yc, zc)

def depth_checks(z):
    print("Z given: ", z)
    if not math.isnan(z) and z > 0.1 and z < 0.6:
        return True
    else:
        return False

class ObjPoseEstimator(object):
    def __init__(self, n_verts=4, tune=False, qr=False):
        self.bridge = CvBridge()
        self.ir_img = None
        self.depth_img = None
        self.depthimage2 = None
        self.rgb_img = None
        self.depth_reg_img = None
        self.rgb_sub = None
        self.dr_sub = None
        self.n_verts = n_verts
        self.tune = tune
        self.qr = qr
        self.fx_d = 0.0
        self.fy_d = 0.0
        self.cx_d = 0.0
        self.cy_d = 0.0
        self.fx_c = 0.0
        self.fy_c = 0.0
        self.cx_c = 0.0
        self.cy_c = 0.0
        self.XYZ = []
        self.verts_best = []
        self.verts_x = None
        self.verts_y = None
        self.cntr = 0
        self.obj_coords_ee = []
        self.once = True
        self.tf_buffer = tf2_ros.Buffer()    
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.initialize_jnts()

    def initialize_jnts(self):
        print("Initializing joints...")
        positions = dict()
        limb = Limb()
        calib_angles = [0.27, -3.27, 3.04, -1.60, -0.38, -1.66, 0.004]
        all_jnts = copy.copy(limb.joint_names())
        limb.set_joint_position_speed(0.2)
        positions['head_pan'] = 0.0

        enum_iter = enumerate(all_jnts, start=0)        
        for i, jnt_name in enum_iter:
            positions[jnt_name] = calib_angles[i]

        limb.move_to_joint_positions(positions)

    def get_QR_coords(self):
        # Read image
        im = self.rgb_img
        
        decodedObjects = pyzbar.decode(im)
        for decodedObject in decodedObjects:
        
            points = decodedObject.polygon
            # If the points do not form a quad, find convex hull
            if len(points) > 4 : 
                hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                hull = list(map(tuple, np.squeeze(hull)))
            else : 
                hull = points;
                
                
            # Number of points in the convex hull
            n = len(hull)
            # Draw the convext hull
            for j in range(0,n):
                cv2.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)
                
            for p in hull:
                if len(self.verts_best) != 4:
                    self.verts_best.append([p.x, p.y])

            print("Best vertices from QR code: ", self.verts_best)
            for verts in self.verts_best:
                im = cv2.circle(im, (verts[0], verts[1]), 2, (255,0,0), 2)
        
            # Display results 
            cv2.imshow("QR coordinates", im)
            cv2.waitKey(3)
        
    def find_vertices_mode(self, vertices):
        if self.verts_x is None:
            self.verts_x = np.zeros((self.n_verts, 100), dtype=int)
            self.verts_y = np.zeros((self.n_verts, 100), dtype=int)
            
        if self.cntr <= 99:
            enum_iter = enumerate(vertices, start=0)
            for i, vert in enum_iter:
                print i, self.cntr, vert
                self.verts_x[i][self.cntr] = vert[0][0]
                self.verts_y[i][self.cntr] = vert[0][1]
            self.cntr += 1
        else:
            self.verts_best = []
            self.XYZ = []
            for i in range(self.n_verts):
                self.verts_best.append([stats.mode(self.verts_x[i])[0][0],
                                        stats.mode(self.verts_y[i])[0][0]])
            print("Best vertices found: ", self.verts_best)
            self.verts_x = None
            self.verts_y = None
            self.cntr = 0
            
    def find_nonNaN_depth(self, verts):
        x = int(verts[0])
        y = int(verts[1])
        
        if depth_checks(self.depthimage2[y][x]):
            return x, y, True

        #Depth is NaN at given vertices, check for nearby points
        search = 10

        for i in range(search):
            if depth_checks(self.depthimage2[y][x+i]):
                return x+i, y, True
            if depth_checks(self.depthimage2[y][x-i]):
                return x-i, y, True
            if depth_checks(self.depthimage2[y+i][x]):
                return x, y+i, True
            if depth_checks(self.depthimage2[y-i][x]):
                return x, y-i, True
            if depth_checks(self.depthimage2[y+i][x+i]):
                return x+i, y+i, True
            if depth_checks(self.depthimage2[y+i][x-i]):
                return x-i, y+i, True
            if depth_checks(self.depthimage2[y-i][x-i]):
                return x-i, y-i, True
            if depth_checks(self.depthimage2[y-i][x+i]):
                return x+i, y-i, True

        return 0, 0, False
    
    def depth_registered_subscr(self, data):
        try:
            #self.depth_reg_img = self.bridge.imgmsg_to_cv2(data, "32FC1")
            self.depth_reg_img = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            self.depthimage2 = np.array(self.depth_reg_img, dtype=np.float32)
        except CvBridgeError as e:
            print("Error :", e)

        if len(self.verts_best) != self.n_verts:
            return


        img = self.rgb_img
        for i in range(self.depthimage2.shape[0]):
            for j in range(self.depthimage2.shape[1]):
                depth = self.depthimage2[i][j]
                if not math.isnan(depth) and depth < 2:
                    #print(i, j, depth)
                    img = cv2.circle(img, (i, j), 2, (0,255,0), 2)

        for verts in self.verts_best:
            img = cv2.circle(img, (verts[0], verts[1]), 2, (255,0,0), 2)
            depth = self.depthimage2[verts[1]][verts[0]]
            print verts[0], verts[1], depth
        cv2.imshow('Points', img)
        cv2.waitKey(3)
    
        verts_post_depthCheck = []
        for verts in self.verts_best:
            x, y, success = self.find_nonNaN_depth(verts)
            if not success:
                print("ERROR: Couldn't find depth for one or more vertices")
                return
            verts_post_depthCheck.append([x,y])
            
        for verts in verts_post_depthCheck:
            x = verts[0]
            y = verts[1]
            Zc = self.depthimage2[int(y)][int(x)]
            Xc = (x - self.cx_d) * Zc / self.fx_d
            Yc = (y - self.cy_d) * Zc / self.fy_d
            print("x, y, Xc, Yc, Zc", x, y, Xc, Yc, Zc)
            Xr, Yr, Zr = transform_camera2robot(Xc, Yc, Zc)
            print("Xr, Yr, Zr", Xr, Yr, Zr)                
            if len(self.XYZ) != len(self.verts_best):
                self.XYZ.append([Xr, Yr, Zr])
        
        if len(self.XYZ) == len(self.verts_best) and len(self.XYZ) != 0:
            print 'Xr, Yr, Zr', self.XYZ
        
    def pcl_subscr(self, data):
        
        if len(self.verts_best) != self.n_verts:
            return
        
        print("pcl_subscr: Best vertices populated!")
        points_list = []
        for p in pc2.read_points(data, skip_nans=True):
            points_list.append([p[0], p[1], p[2], p[3]])
            print p[0], p[1], p[2], p[3]
        pcl_data = pcl.PointCloud_PointXYZRGB()
        pcl_data.from_list(points_list)
        '''
        for pd in pcl_data:
            print pd
        width = 640
        for vert in self.verts_best:
            x_pixel = vert[0]
            y_pixel = vert[1]
            Xc = pcl_data.point[width * y_pixel + x_pixel].x
            Yc = pcl_data.point[width * y_pixel + x_pixel].y;
            Zc = pcl_data.point[width * y_pixel + x_pixel].z;
            Xr, Yr, Zr = transform_camera2robot(Xc, Yc, Zc)
            print("Xr, Yr, Zr", Xr, Yr, Zr)                
            if len(self.XYZ) != len(self.verts_best):
                self.XYZ.append([Xr, Yr, Zr])
        '''
                
    def rgb_image_subscr(self, data):
        try:
            self.rgb_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print("Error :", e)

        if self.qr:
            self.get_QR_coords()
            return
    
        if self.tune:
            # create trackbars for color change experimentation
            cv2.namedWindow('Converted Image')
            cv2.createTrackbar('Lower Hue','Converted Image',0,180,nothing)
            cv2.createTrackbar('Lower Sat','Converted Image',0,255,nothing)
            cv2.createTrackbar('Lower Value','Converted Image',0,255,nothing)
            cv2.createTrackbar('Upper Hue','Converted Image',0,180,nothing)
            cv2.createTrackbar('Upper Sat','Converted Image',0,255,nothing)
            cv2.createTrackbar('Upper Value','Converted Image',0,255,nothing)
            
            lowh = cv2.getTrackbarPos('Lower Hue','Converted Image')
            lows = cv2.getTrackbarPos('Lower Sat','Converted Image')
            lowv = cv2.getTrackbarPos('Lower Value','Converted Image')
            upph = cv2.getTrackbarPos('Upper Hue','Converted Image')
            upps = cv2.getTrackbarPos('Upper Sat','Converted Image')
            uppv= cv2.getTrackbarPos('Upper Value','Converted Image')
            switch = '0 : OFF \n1 : ON'
            cv2.createTrackbar(switch, 'Converted Image',0,1,nothing)
            lower = np.array([lowh,lows,lowv])
            upper = np.array([upph,upps,uppv])
        else:
            # values found through experimentation for blue color
            lower = np.array([75, 160, 115])
            upper = np.array([115, 255, 255])
        
        hsv = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(self.rgb_img, self.rgb_img, mask=mask)

        if self.tune:
            cv2.imshow("Side by side Image", np.hstack([self.rgb_img, res]))
            cv2.waitKey(100)
        
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(contours) <= 0:
            return
        cnt = max(contours, key = cv2.contourArea)
        hull = cv2.convexHull(cnt, clockwise=True)
        perimeter = 0.01*cv2.arcLength(cnt, True)
        vertices = cv2.approxPolyDP(hull, perimeter, True)
        
        if len(vertices) == self.n_verts:
            self.find_vertices_mode(vertices)
        cv2.drawContours(self.rgb_img, [hull], -1, (0, 255, 0), 3)
        cv2.imshow('Contour', self.rgb_img)
        cv2.waitKey(3)

    def ee_pose(self, msg):
        '''
        Dont do the transform if the XYZ coordinates of the objects 
        (in robot base) are not yet populated 
        Or it is possible that we have already calculated the coordinates
        and no more processing is needed
        '''
        if len(self.XYZ) == 0 or len(self.obj_coords_ee) != 0:
            return
        
        try:
            # We want to transform from robot base to right_gripper_base
            tform = self.tf_buffer.lookup_transform("right_gripper_base",
                                                    "base",
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

        for coords in self.XYZ:
            obj_pose = PoseStamped()
            obj_pose.header.frame_id = "right_gripper_base"
            obj_pose.pose.position.x = coords[0]
            obj_pose.pose.position.y = coords[1]
            obj_pose.pose.position.z = coords[2]
            pose44 = np.dot(xyz_to_mat44(obj_pose.pose.position),
                            xyzw_to_mat44(obj_pose.pose.orientation))
            txpose = np.dot(mat44, pose44)
            xyz = tuple(transformations.translation_from_matrix(txpose))[:3]
            x, y, z = xyz
            self.obj_coords_ee.append([x,y,z])
            print(x,y,z)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verts', type=int)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--qr', action='store_true')
    args = parser.parse_args()

    rospy.init_node('obj_pose_estimator')
    print("initialized obj_pose_estimator ....")
    obs = ObjPoseEstimator(n_verts=args.verts, tune=args.tune, qr=args.qr)
    
    # Get the camera calibration parameter for the rectified image
    #     [fx'  0  cx' Tx]
    # P = [ 0  fy' cy' Ty]
    #     [ 0   0   1   0]
    msg = rospy.wait_for_message('/camera/depth/camera_info', CameraInfo, timeout=None) 
    obs.fx_d = msg.P[0]
    obs.fy_d = msg.P[5]
    obs.cx_d = msg.P[2]
    obs.cy_d = msg.P[6]
    print "Depth Camera Info", obs.fx_d, obs.fy_d, obs.cx_d, obs.cy_d
    
    msg = rospy.wait_for_message('/camera/rgb/camera_info', CameraInfo, timeout=None) 
    obs.fx_c = msg.P[0]
    obs.fy_c = msg.P[5]
    obs.cx_c = msg.P[2]
    obs.cy_c = msg.P[6]
    print "RGB Camera Info", obs.fx_c, obs.fy_c, obs.cx_c, obs.cy_c
    
    try:
        
        obs.rgb_sub = rospy.Subscriber("/camera/rgb/image_raw",
                                       Image, obs.rgb_image_subscr)

        obs.dr_sub = rospy.Subscriber("/camera/depth_registered/image_raw",
                                      Image, obs.depth_registered_subscr)
        '''
        obs.pcl_sub = rospy.Subscriber("/camera/depth_registered/points",
                                       PointCloud2, obs.pcl_subscr)
        '''
        obs.ee_sub = rospy.Subscriber('/robot/limb/right/endpoint_state',
                                      EndpointState,
                                      obs.ee_pose,
                                      queue_size=1)
        print "subscribed to topics ...."
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()
  
if __name__ == "__main__":
    main()
