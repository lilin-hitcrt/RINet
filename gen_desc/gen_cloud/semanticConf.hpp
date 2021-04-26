#pragma once
#include <yaml-cpp/yaml.h>
#include <string>
#include "types.hpp"
typedef std::tuple<u_char, u_char, u_char> Color;
    class semConf
    {
    private:
        std::map<uint32_t, Color> _color_map, _argmax_to_rgb;
        YAML::Node learning_map;
        std::vector<int> label_map;
        bool remap_label = true;
        semConf();

    public:
        semConf(std::string conf_file);
        ~semConf() = default;
        int remap(uint32_t in_label);
        Color getColor(uint32_t label);
        CloudCPtr getColorCloud(CloudLPtr &cloud_in);
    };
