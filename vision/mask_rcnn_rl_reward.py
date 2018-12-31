#!/usr/bin/env python
import cv2 as cv
import argparse
import numpy as np
import os.path
import sys
import random
import rospy
import roslib
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class MaskRCNNReward(object):
    def __init__(self):
        
        # Initialize the parameters
        self.confThreshold = 0.5  # Confidence threshold
        self.maskThreshold = 0.3  # Mask threshold
        self.bridge = CvBridge()
        self.rgb_img = None
        
        # Load names of classes
        classesFile = "./mask_rcnn_inception_v2_coco_2018_01_28/mscoco_labels.names"
        self.classes = None
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        # Give the textGraph and weight files for the model
        textGraph = "./mask_rcnn_inception_v2_coco_2018_01_28/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
        modelWeights = "./mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
        
        # Load the network
        self.net = cv.dnn.readNetFromTensorflow(modelWeights,
                                                textGraph);
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        
        # Load the classes
        colorsFile = "./mask_rcnn_inception_v2_coco_2018_01_28/colors.txt";
        with open(colorsFile, 'rt') as f:
            colorsStr = f.read().rstrip('\n').split('\n')
        self.colors = [] #[0,0,0]

        for i in range(len(colorsStr)):
            rgb = colorsStr[i].split(' ')
            color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
            self.colors.append(color)

    # Draw the predicted bounding box, colorize and show the mask on the image
    def drawBox(self, frame, classId, conf, left, top,
                right, bottom, classMask):

        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
        # Print a label of class.
        label = '%.2f' % conf
        if classes:
            assert(classId < len(classes))
            label = '%s:%s' % (classes[classId], label)
        
        # Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - int(round(1.5*labelSize[1]))), (left + int(round(1.5*labelSize[0])), top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
        
        # Resize the mask, threshold, color and apply it on the image
        classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
        mask = (classMask > self.maskThreshold)
        roi = frame[top:bottom+1, left:right+1][mask]

        # color = colors[classId%len(colors)]
        # Comment the above line and uncomment the two lines below to generate different instance colors
        colorIndex = random.randint(0, len(colors)-1)
        color = self.colors[colorIndex]

        frame[top:bottom+1, left:right+1][mask] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.7 * roi).astype(np.uint8)
        
        # Draw the contours on the image
        mask = mask.astype(np.uint8)
        im2, contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(frame[top:bottom+1, left:right+1], contours, -1, color, 3, cv.LINE_8, hierarchy, 100)

    # For each frame, extract the bounding box and mask for each detected object
    def postprocess(self, boxes, masks):
        
        # Output size of masks is NxCxHxW where
        # N - number of detected boxes
        # C - number of classes (excluding background)
        # HxW - segmentation shape
        numClasses = masks.shape[1]
        numDetections = boxes.shape[2]

        frameH = self.rgb_img.shape[0]
        frameW = self.rgb_img.shape[1]
        
        for i in range(numDetections):
            box = boxes[0, 0, i]
            mask = masks[i]
            score = box[2]
            if score > self.confThreshold:
                classId = int(box[1])
                
                # Extract the bounding box
                left = int(frameW * box[3])
                top = int(frameH * box[4])
                right = int(frameW * box[5])
                bottom = int(frameH * box[6])
                
                left = max(0, min(left, frameW - 1))
                top = max(0, min(top, frameH - 1))
                right = max(0, min(right, frameW - 1))
                bottom = max(0, min(bottom, frameH - 1))
                
                # Extract the mask for the object
                classMask = mask[classId]
            
                # Draw bounding box, colorize and show the mask on the image
                self.drawBox(self.rgb_img, classId, score, left, top,
                             right, bottom, classMask)

    # RGB Image Msg callback
    def rgb_image_subscr(self, data):
        try:
            self.rgb_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print("Error :", e)
        
        while cv.waitKey(1) < 0:
    
            # Create a 4D blob from a frame.
            blob = cv.dnn.blobFromImage(self.rgb_img, swapRB=True, crop=False)

            # Set the input to the network
            self.net.setInput(blob)

            # Run the forward pass to get output from the output layers
            boxes, masks = self.net.forward(['detection_out_final',
                                             'detection_masks'])

            # Extract the bounding box and mask for each of the detected objects
            self.postprocess(boxes, masks)

            # Put efficiency information.
            t, _ = self.net.getPerfProfile()
            label = 'Mask-RCNN on 2.5 GHz Intel Core i7 CPU, Inference time for a frame : %0.0f ms' % abs(t * 1000.0 / cv.getTickFrequency())
            print label
            
            winName = 'Mask-RCNN Object detection and Segmentation in OpenCV'
            cv.namedWindow(winName, cv.WINDOW_NORMAL)
            cv.imshow(winName, self.rgb_img)
    
def main():
    rospy.init_node('mask_rcnn_reward')
    print("Initialized mask_rcnn_reward...")
    rew = MaskRCNNReward()
    
    try: 
        rew.rgb_sub = rospy.Subscriber("/camera/rgb/image_raw",
                                       Image, rew.rgb_image_subscr)

        print "Subscribed to topics!"
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()
  
if __name__ == "__main__":
    main()

    
