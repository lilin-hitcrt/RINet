#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <opencv2/opencv.hpp>
#include "semanticConf.hpp"
#include "genData.hpp"

int main(int argvc,char** argv){
    std::string cloud_path=argv[1];
    std::string label_path=argv[2];
    bool label_valid[20]={0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1};
    bool use_min[20]={1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1};
    int label_map[20]={-1,0,-1,-1,-1,-1,-1,-1,-1,1,2,3,4,5,6,7,8,9,10,11};
	std::shared_ptr<semConf> semconf(new semConf("../conf/sem_config.yaml"));
	genData gener(cloud_path,label_path, semconf);
    CloudLPtr cloud(new CloudL);
    int totaldata = gener.totaldata;
    int num=0;
    pcl::visualization::CloudViewer viewer("cloud");
    std::ofstream fout(argv[3],ios::binary);
    while (gener.getData(cloud)){
        std::cout<<num<<"/"<<totaldata<<std::endl;
        CloudLPtr cloud_out(new CloudL);
        std::vector<float> dis_list;
        cloud_out->resize((label_map[19]+1)*360);
        dis_list.resize(cloud_out->size(),0.f);
        for(auto p:cloud->points){
            if(label_valid[p.label]){
                int angle=std::floor((std::atan2(p.y,p.x)+M_PI)*180./M_PI);
                if(angle<0||angle>359){
                    continue;
                }
                float dis=std::sqrt(p.x*p.x+p.y*p.y);
                if(dis>50){
                    continue;
                }
                auto& q=cloud_out->at(360*label_map[p.label]+angle);
                if(q.label>0){
                    float dis_temp=std::sqrt(q.x*q.x+q.y*q.y);
                    if(use_min[p.label]){
                        if(dis<dis_temp){
                            q=p;
                            dis_list[360*label_map[p.label]+angle]=dis;
                        }
                    }else{
                        if(dis>dis_temp){
                            q=p;
                            dis_list[360*label_map[p.label]+angle]=dis;
                        }
                    }
                }else{
                    q=p;
                    dis_list[360*label_map[p.label]+angle]=dis;
                }
            }
        }
        for(auto dis:dis_list){
            fout.write((char*)(&dis),sizeof(dis));
        }
        auto ccloud=semconf->getColorCloud(cloud_out);
        viewer.showCloud(ccloud);
        ++num;
    }
    fout.close();
    return 0;
}