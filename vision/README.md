# MSR Capstone Project - Computer Vision pipeline for RL algorithm experiments
Srikanth Kilaru

Northwestern University (Fall 2018)

## Overview
This README covers the algorithm and software implementation aspects of the Computer Vision portion of my capstone project.
Please see my [portfolio page](https://srikanth-kilaru.github.io/projects/2018/final-proj-RL) for more details about the project.
The Computer Vision portion of the RL experiments was implemented to estimate the pose of the object in the gripper and the pose of the goal. These two poses are part of the observations used as input to the RL algorithms (PG and Actor-Critic). For an RL algorithm to succeed, the learned policy should move the object in the gripper to the goal, in other words, the pose of the object and the pose of the goal should match.
Cartesian coordinates of 3 points on the out facing surface of the object are used to identify the pose of an object.

## Algorithm

### Object pose identification
Before the the RL training phase begins, the ROS node in obj_pose_estimate.py script is used to estimate the pose of the object held by the gripper in the gripper's reference frame. This pose stays fixed throughout the experiment as the object does not move in the gripper reference frame. However, as the end effector moves through space, this fixed object pose in the gripper's frame, is changing in Sawyer's base frame. The object's pose in the gripper frame is transformed into a pose in Sawyer's base frame by using the transforms published by the tf service. This transformation is necessary to compute the distance between the goal and the object in a common reference frame, for use by the RL algorithm's reward function.


During this pre-training phase, a QR code is attached to the outward facing surface of the object to enable detection of corners of the QR code. 
The pose estimation script moves the robot arm to a location which is well within the sight of the fixed camera, located on top of Sawyer's head in this experiment, to estimate the X, Y, and Z coordinates of the corners of the QR code, in the camera frame. A calibrated ASUS XtionPRO LIVE RGB-D camera is used to estimate these coordinates.
The object pose estimation algorithm follows these steps -

* Subscribe to:

/camera/depth/camera_info

/camera/depth_registered/image_raw

/camera/rgb/camera_info

/camera/rgb/image_raw

/robot/limb/right/endpoint_state

* RGB image is used to calculate the QR code corners in pixel space. QR corner estimation code snippet is attributed to the [Learn OpenCV web site](https://github.com/spmallick/learnopencv)

* Depth registered image is used to estimate the depth of these corner pixels.
NOTE: Invalid depth values are eliminated.

* Transform the coordinates of the three points on the object from Camera frame to Sawyer's base frame. This should be a known tranformation based upon where the camera reference frame is located in relation to Sawyer's base reference frame. These coordinates are then transformed to the Endpoint / Gripper frame in the callback to the ROS topic /robot/limb/right/endpoint_state using the tf service.

* The coordinates of these three points are then written to the file specified to the script, the init.yaml file used by the RL algorithms during training and testing phase.

### Goal pose estimation

As explained earlier, the pose of the goal is required as an observation during the training phase. It is also required during the testing phase of the RL algorithm as the policy execution is stopped when the observation of the object's pose following a policy action matches the pose of the goal specified in the testing phase.

For estimation of goal poses, the RGB-D camera points towards the shape sorting block (goal) and identifies in the RGB image the vertices of the shape specified as the goal by using contour matching logic available in OpenCV.
To assist with noise elimination due to lighting conditions and the pale color of the wood used in making the shape sorting block, a dark blue color paper was inserted below the shape sorting holes to assist in contour detection.

The goal pose estimation algorithm follows these steps -

* Subscribe to:

/camera/depth/camera_info

/camera/depth_registered/image_raw

/camera/rgb/camera_info

/camera/rgb/image_raw

* RGB image is used to find three vertices of the selected shape in pixel space. 

* Depth registered image is used to estimate the depth of these vertex pixels.
NOTE: Invalid depth values are eliminated.

* Transform the coordinates of the three points on the object from Camera frame to Sawyer's base frame. This should be a known tranformation based upon where the camera reference frame is located in relation to Sawyer's base reference frame. 

* The coordinates of these three points on the goal are then written to the file specified to the script, the init.yaml file used by the RL algorithms during training and testing phase.