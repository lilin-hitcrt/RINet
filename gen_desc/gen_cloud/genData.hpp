#pragma once
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>
#include <pcl/visualization/cloud_viewer.h>
#include <opencv2/opencv.hpp>
#include "semanticConf.hpp"
#include "types.hpp"
    class genData
    {
    private:
        CloudLPtr getLCloud(std::string file_cloud, std::string file_label);
        CloudLPtr getCloud();
        std::vector<std::string> listDir(std::string path, std::string end);
        std::vector<std::string> split(const std::string& str, const std::string& delim);
        std::string cloud_path,label_path;
        std::vector<std::string> label_filenames;
        std::shared_ptr<semConf> semconf;
        std::shared_ptr<pcl::visualization::CloudViewer> viewer;
        int data_id=0;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int totaldata = 0;
        genData(std::string cloud_path,std::string label_path,std::shared_ptr<semConf> semconf);
        bool getData(CloudLPtr &cloud);
        ~genData()=default;
    };
