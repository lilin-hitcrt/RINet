#include "genData.hpp"
genData::genData(std::string _cloud_path, std::string _label_path, std::shared_ptr<semConf> _semconf)
{
    this->semconf = _semconf;
    cloud_path = _cloud_path;
    label_path = _label_path;
    label_filenames = listDir(label_path, ".label");
    totaldata = label_filenames.size();
}
std::vector<std::string> genData::listDir(std::string path, std::string end)
{
    DIR *pDir;
    struct dirent *ptr;
    std::vector<std::string> files;
    if (!(pDir = opendir(path.c_str())))
    {
        return files;
    }
    std::string subFile;
    while ((ptr = readdir(pDir)) != 0)
    {
        subFile = ptr->d_name;
        auto rt = subFile.find(end);
        if (rt != std::string::npos)
        {
            files.emplace_back(path + subFile);
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::vector<std::string> genData::split(const std::string& str, const std::string& delim){  
	std::vector<std::string> res;  
	if("" == str) return res;   
	char * strs = new char[str.length() + 1] ;
	strcpy(strs, str.c_str());   
	char * d = new char[delim.length() + 1];  
	strcpy(d, delim.c_str());  
	char *p = strtok(strs, d);  
	while(p) {  
		std::string s = p;  
		res.push_back(s); 
		p = strtok(NULL, d);  
	}  
	return res;  
}

CloudLPtr genData::getCloud()
{
    auto cloud_file=split(split(label_filenames[data_id],"/").back(),".")[0]+".bin";
    return getLCloud(cloud_path+cloud_file, label_filenames[data_id]);
}
CloudLPtr genData::getLCloud(std::string file_cloud, std::string file_label)
{
    CloudLPtr re_cloud(new CloudL);
    std::ifstream in_label(file_label, std::ios::binary);
    if (!in_label.is_open())
    {
        std::cerr << "No file:" << file_label << std::endl;
        exit(-1);
    }
    in_label.seekg(0, std::ios::end);
    uint32_t num_points = in_label.tellg() / sizeof(uint32_t);
    in_label.seekg(0, std::ios::beg);
    std::vector<uint32_t> values_label(num_points);
    in_label.read((char *)&values_label[0], num_points * sizeof(uint32_t));
    std::ifstream in_cloud(file_cloud, std::ios::binary);
    std::vector<float> values_cloud(4 * num_points);
    in_cloud.read((char *)&values_cloud[0], 4 * num_points * sizeof(float));
    re_cloud->points.resize(num_points);
    for (uint32_t i = 0; i < num_points; ++i)
    {
        uint32_t sem_label;
        sem_label = semconf->remap(values_label[i]);
        re_cloud->points[i].x = values_cloud[4 * i];
        re_cloud->points[i].y = values_cloud[4 * i + 1];
        re_cloud->points[i].z = values_cloud[4 * i + 2];
        re_cloud->points[i].label = sem_label;
    }
    in_label.close();
    in_cloud.close();
    return re_cloud;
}

bool genData::getData(CloudLPtr &cloud)
{
    if (data_id >= totaldata)
    {
        return false;
    }
    if (cloud == NULL)
    {
        cloud.reset(new CloudL);
    }
    auto label_file=label_filenames[data_id];
    cloud = getCloud();
    data_id++;
    return true;
}
