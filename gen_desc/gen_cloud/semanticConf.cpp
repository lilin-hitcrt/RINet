#include "semanticConf.hpp"

semConf::semConf(std::string conf_file)
{
    auto data_cfg = YAML::LoadFile(conf_file);
    remap_label = data_cfg["remap"].as<bool>();
    auto color_map = data_cfg["color_map"];
    learning_map = data_cfg["learning_map"];
    label_map.resize(260);
    for (auto it = learning_map.begin(); it != learning_map.end(); ++it)
    {
        label_map[it->first.as<int>()] = it->second.as<int>();
    }
    YAML::const_iterator it;
    for (it = color_map.begin(); it != color_map.end(); ++it)
    {
        // Get label and key
        int key = it->first.as<int>(); // <- key
        Color color = std::make_tuple(
            static_cast<u_char>(color_map[key][0].as<unsigned int>()),
            static_cast<u_char>(color_map[key][1].as<unsigned int>()),
            static_cast<u_char>(color_map[key][2].as<unsigned int>()));
        _color_map[key] = color;
    }
    auto learning_class = data_cfg["learning_map_inv"];
    for (it = learning_class.begin(); it != learning_class.end(); ++it)
    {
        int key = it->first.as<int>(); // <- key
        _argmax_to_rgb[key] = _color_map[learning_class[key].as<unsigned int>()];
    }
}

int semConf::remap(uint32_t in_label)
{
    if (remap_label)
    {
        return label_map[(int)(in_label & 0x0000ffff)];
    }
    else
    {
        return in_label;
    }
}

Color semConf::getColor(uint32_t label)
{
    return _argmax_to_rgb[label];
}
CloudCPtr semConf::getColorCloud(CloudLPtr &cloud_in)
{
    CloudCPtr outcloud(new CloudC);
    outcloud->points.resize(cloud_in->points.size());
    for (size_t i = 0; i < outcloud->points.size(); i++)
    {
        outcloud->points[i].x = cloud_in->points[i].x;
        outcloud->points[i].y = cloud_in->points[i].y;
        outcloud->points[i].z = cloud_in->points[i].z;
        auto color = getColor(cloud_in->points[i].label);
        outcloud->points[i].r = std::get<0>(color);
        outcloud->points[i].g = std::get<1>(color);
        outcloud->points[i].b = std::get<2>(color);
    }
    outcloud->height = 1;
    outcloud->width = outcloud->points.size();
    return outcloud;
}