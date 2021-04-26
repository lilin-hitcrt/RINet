#pragma once
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
typedef pcl::PointXYZL PointL;
typedef pcl::PointCloud<PointL> CloudL;
typedef CloudL::Ptr CloudLPtr;

typedef pcl::PointXYZ Point;
typedef pcl::PointCloud<Point> Cloud;
typedef Cloud::Ptr CloudPtr;

typedef pcl::PointXYZRGB PointC;
typedef pcl::PointCloud<PointC> CloudC;
typedef CloudC::Ptr CloudCPtr;
