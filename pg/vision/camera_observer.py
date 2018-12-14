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

def nothing(x):
    pass

def getXYZ(xp, yp, zc, fx, fy, cx, cy):
    xn = (xp - cx) / fx
    yn = (yp - cy) / fy
    xc = xn * zc
    yc = yn * zc
    return (xc, yc, zc)

class CameraObserver(object):
    def __init__(self, n_verts=4):
        self.bridge = CvBridge()
        self.ir_img = None
        self.depth_img = None
        self.depthimage2 = None
        self.rgb_img = None
        self.depth_reg_img = None
        self.rgb_sub = None
        self.dr_sub = None
        self.n_verts = n_verts
        
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
        
    def find_vertices_mode(self, vertices):
        if len(vertices) != self.n_verts:
            return
        
        if self.verts_x is None:
            self.verts_x = np.zeros((self.n_verts, 100), dtype=int)
            self.verts_y = np.zeros((self.n_verts, 100), dtype=int)
            
        if self.cntr <= 99:
            enum_iter = enumerate(vertices, start=0)
            for i, vert in enum_iter:
                #print i, self.cntr, vert
                self.verts_x[i][self.cntr] = vert[0][0]
                self.verts_y[i][self.cntr] = vert[0][1]
            self.cntr += 1
        else:
            self.verts_best = []
            for i in range(self.n_verts):
                self.verts_best.append([stats.mode(self.verts_x[i])[0][0],
                                        stats.mode(self.verts_y[i])[0][0]])
            print self.verts_best
            self.verts_x = None
            self.verts_y = None
            self.cntr = 0
            

    def depth_registered_subscr(self, data):
        try:
            self.depth_reg_img = self.bridge.imgmsg_to_cv2(data, "32FC1")
            depthimage2 = np.array(self.depth_reg_img, dtype=np.float32)
        except CvBridgeError as e:
            print("Error :", e)

        for verts in self.verts_best:
            x = verts[0]
            y = verts[1]
            if not math.isnan(depthimage2[int(y)][int(x)]) and depthimage2[int(y)][int(x)] > 0.1 and depthimage2[int(y)][int(x)] < 10.0:
                Z = depthimage2[int(y)][int(x)]
                X = (x - self.cx_d) * Z / self.fx_d
                Y = (y - self.cy_d) * Z / self.fy_d
                if len(self.XYZ) != len(self.verts_best):
                    self.XYZ.append([X, Y, Z])
        
        if len(self.XYZ) == len(self.verts_best) and len(self.verts_best) != 0:
            print 'xc, yc, zc', self.XYZ

        
    def rgb_image_subscr(self, data):
        try:
            self.rgb_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print("Error :", e)

        '''
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
        '''

        # values found through experimentation for blue color
        lower = np.array([110,175,0])
        upper = np.array([255,255,255])
        
        hsv = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(self.rgb_img, self.rgb_img, mask=mask)
        
        '''
        cv2.imshow("Side by side Image", np.hstack([self.rgb_img, res]))
        cv2.waitKey(100)
        '''
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(contours) > 0:
            cnt = max(contours, key = cv2.contourArea)
            hull = cv2.convexHull(cnt, clockwise=True)
            perimeter = 0.01*cv2.arcLength(cnt, True)
            vertices = cv2.approxPolyDP(hull, perimeter, True)
            # change this later to account for user specified shape
            if len(vertices) == self.n_verts:
                self.find_vertices_mode(vertices)
            cv2.drawContours(self.rgb_img, [hull], -1, (0, 255, 0), 3)
            cv2.imshow('draw contours', self.rgb_img)
            cv2.waitKey(3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verts', type=int)
    args = parser.parse_args()
    verts = args.verts
    
    obs = CameraObserver(n_verts=verts)
    print "trying to initialize camera_observer ...."
    rospy.init_node('cam_obs')
    print "initialized camera_observer ...."

    # Get the camera calibration parameter for the rectified image
    #     [fx'  0  cx' Tx]
    # P = [ 0  fy' cy' Ty]
    #     [ 0   0   1   0]
    msg = rospy.wait_for_message('/camera/depth/camera_info', CameraInfo, timeout=None) 
    obs.fx_d = msg.P[0]
    obs.fy_d = msg.P[5]
    obs.cx_d = msg.P[2]
    obs.cy_d = msg.P[6]
    print obs.fx_d, obs.fy_d, obs.cx_d, obs.cy_d

    msg = rospy.wait_for_message('/camera/rgb/camera_info', CameraInfo, timeout=None) 
    obs.fx_c = msg.P[0]
    obs.fy_c = msg.P[5]
    obs.cx_c = msg.P[2]
    obs.cy_c = msg.P[6]
    print obs.fx_c, obs.fy_c, obs.cx_c, obs.cy_c
    
    try:
        obs.rgb_sub = rospy.Subscriber("/camera/rgb/image_raw",
                                       Image, obs.rgb_image_subscr)
        obs.dr_sub = rospy.Subscriber("/camera/depth_registered/image_raw",
                                      Image, obs.depth_registered_subscr)

        print "subscribed to topics ...."
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()
  
if __name__ == "__main__":
    main()
