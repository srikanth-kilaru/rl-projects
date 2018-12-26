#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>
#include </opt/ros/kinetic/include/cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <zbar.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <math.h>       /* sin */

#define PI 3.14159265

using namespace cv;
using namespace std;
using namespace zbar;


// Set resolution
#define WIDTH  640
#define HEIGHT 480

typedef struct
{
  string type;
  string data;
  vector <cv::Point> location;
} decodedObject;

int x_pixel[4];
int y_pixel[4];
double Xc[4];
double Yc[4];
double Zc[4];

// Find and decode barcodes and QR codes
void decode(cv::Mat &im, vector<decodedObject>&decodedObjects)
{
  // Create zbar scanner
  ImageScanner scanner;
 
  // Configure scanner
  scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
   
  // Convert image to grayscale
  Mat imGray;
  cvtColor(im, imGray,CV_BGR2GRAY);
 
  // Wrap image data in a zbar image
  Image image(im.cols, im.rows, "Y800", (uchar *)imGray.data, im.cols * im.rows);
 
  // Scan the image for barcodes and QRCodes
  int n = scanner.scan(image);
   
  // Print results
  for(Image::SymbolIterator symbol = image.symbol_begin(); symbol != image.symbol_end(); ++symbol)
  {
    decodedObject obj;
     
    obj.type = symbol->get_type_name();
    obj.data = symbol->get_data();
     
    // Print type and data
    //cout << "Type : " << obj.type << endl;
    //cout << "Data : " << obj.data << endl;
     
    // Obtain location
    for(int i = 0; i< symbol->get_location_size(); i++)
    {
      obj.location.push_back(Point(symbol->get_location_x(i),symbol->get_location_y(i)));
      x_pixel[i] = symbol->get_location_x(i);
      y_pixel[i] = symbol->get_location_y(i);
      //cout << "x_pixel: " << x_pixel[i] << endl;
      //cout << "y_pixel: " << y_pixel[i] << endl;
    }
     
    decodedObjects.push_back(obj);
  }
}


// Display barcode and QR code location  
void display(Mat &im, vector<decodedObject>&decodedObjects)
{
  // Loop over all decoded objects
  for(int i = 0; i < decodedObjects.size(); i++)
  {
    vector<Point> points = decodedObjects[i].location;
    vector<Point> hull;
     
    // If the points do not form a quad, find convex hull
    if(points.size() > 4)
      convexHull(points, hull);
    else
      hull = points;
     
    // Number of points in the convex hull
    int n = hull.size();
     
    for(int j = 0; j < n; j++)
    {
      line(im, hull[j], hull[ (j+1) % n], Scalar(255,0,0), 3);
    }
     
  }
   
  // Display results 
  imshow("Results", im);
  waitKey(3);
   
}

cv::Mat transform_camera2robot(double X, double Y, double Z)
{
  cv::Vec<double, 4> obj_loc_kin(X, Y, Z, 1);
  cv::Vec<double, 4> kinect_loc(0.0, 0.0, 0.6604, 1);
   
  double rotZ[4][4] = {{cos(PI), -sin(PI), 0, 0}, {sin(PI), cos(PI), 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  cv::Mat rotZ_m = cv::Mat(4, 4, CV_64FC1, rotZ);
  double rotX[4][4] = {{1, 0, 0, 0}, {0, cos(PI/2), -sin(PI/2), 0}, {0, sin(PI/2), cos(PI/2), 0}, {0, 0, 0, 1}};
  cv::Mat rotX_m = cv::Mat(4, 4, CV_64FC1, rotX);
  cv::Mat R = rotX_m.mul(rotZ_m);

  //do the transformation
  cv::Mat kinect_to_sawyer = R * cv::Mat(obj_loc_kin);    
  cv::Mat sawyer_to_obj = kinect_to_sawyer +  cv::Mat(kinect_loc);
  
  return (sawyer_to_obj);
}
  
void pcl_imageCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  //ROS_INFO("Cloud Time Stamp: %f", cloud_msg->header.stamp.toSec());

  pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PCLPointCloud2 pcl_pc2;
  pcl_conversions::toPCL(*cloud_msg, pcl_pc2);
  pcl::fromPCLPointCloud2(pcl_pc2, *temp_cloud);
 
  //pcl::fromROSMsg(*cloud_msg, *temp_cloud);

  int i;
  int good_cnt = 0;
  std::stringstream buffer;
  for (i = 0; i < 4; i++) {
    Xc[i] = temp_cloud->points[WIDTH * y_pixel[i] + x_pixel[i]].x;
    Yc[i] = temp_cloud->points[WIDTH * y_pixel[i] + x_pixel[i]].y;
    Zc[i] = temp_cloud->points[WIDTH * y_pixel[i] + x_pixel[i]].z;
    buffer = std::stringstream();
    buffer << "Xc, Yc, Zc is: ";
    buffer << "(" << Xc[i] << ", " << Yc[i] << ", " << Zc[i] << ")";
    buffer << "for x_pixel, y_pixel: ";
    buffer << "(" << x_pixel[i] << ", " << y_pixel[i] << ")" << endl;
    cout << buffer.str();
    if (!isnan(Xc[i]) && !isnan(Yc[i]) && !isnan(Zc[i]))
      good_cnt++;
    if (good_cnt >= 3) {
      cout << "found good coordinates!!!!!!!!" << good_cnt << endl;
      ofstream filep;
      int k;
      cv::Mat robot_coords;
      
      filep.open("good-coords.txt");
      for (k = 0; k < 4; k++) {
	if (!isnan(Xc[k]) && !isnan(Yc[k]) && !isnan(Zc[k])) {
	  robot_coords = transform_camera2robot(Xc[k], Yc[k], Zc[k]);
	  //filep << Xc[k] << ", " << Yc[k] << ", " << Zc[k] << endl;
	  filep << robot_coords.at<double>(0,1) << ", " << robot_coords.at<double>(1,1) << ", " << robot_coords.at<double>(2,1) << endl;
	}
      }
      filep.close();
    }
  }
}

void rgb_imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  //ROS_INFO("Image Time Stamp: %f", msg->header.stamp.toSec()); 
  try
  {
    // Read image
    cv::Mat im = cv_bridge::toCvShare(msg, "bgr8")->image;
    
    // Variable for decoded objects 
    vector<decodedObject> decodedObjects;
    
    // Find and decode barcodes and QR codes
    decode(im, decodedObjects);
    
    // Display location 
    display(im, decodedObjects);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

int main(int argc, char* argv[])
{
  // Initialize ROS
  ros::init(argc, argv, "obj_pose_estimation_node");
  ROS_INFO("Initialized obj_pose_estimation_node");

  // Declare Node Handle
  ros::NodeHandle nh("~");

  ros::Subscriber rgb_sub = nh.subscribe("/camera/rgb/image_rect_color",
					 1, rgb_imageCallback);
  
  ros::Subscriber pcl_sub = nh.subscribe("/camera/depth_registered/points",
					 1, pcl_imageCallback);

  // Spin Forever
  ros::spin();
  
  return EXIT_SUCCESS;
}


