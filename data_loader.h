#pragma once 
#include <iostream>
#include <vector>
#include <tuple>
// #include "load_txt.h"
#include <torch/torch.h>

class CustomDatasetBatchFileCycles: public torch::data::Dataset<CustomDatasetBatchFileCycles>
{
    private:
        std::vector<std::tuple<vec2f, float>> data;
    public:
        explicit CustomDatasetBatchFileCycles(vec2f seed, std::vector<float> file_cycles)
        {
            for(int i = 0; i < file_cycles.size(); i++){
                data.push_back(std::make_tuple(seed, file_cycles[i]));
            }
        };
        // Override the get method to load custom data.
        torch::data::Example<> get(size_t index) override {

            vec2f start = std::get<0>(data[index]);
            float t = std::get<1>(data[index]);
            torch::Tensor start_tensor = torch::tensor({start.x, start.y});
            auto start_tensor_reshape = start_tensor.view({1, 3});
            torch::Tensor t_tensor = torch::tensor({t});
            auto t_tensor_reshape = t_tensor.view({1, 1});

            return {start_tensor, t_tensor};
        };

        // Override the size method to infer the size of the data set.
        torch::optional<size_t> size() const override {
            return data.size();
        };

};

class CustomDataset: public torch::data::Dataset<CustomDataset>
{
    private:
        std::vector<std::tuple<vec2f, float>> data;
    public:
        explicit CustomDataset(std::vector<vec2f> seeds, std::vector<float> file_cycles, vec2f bbox_lower, vec2f bbox_upper)
        {
            float minval = -1.f;
            float maxval = 1.f;
            for(int s = 0; s < seeds.size(); ++s){
                vec2f seed = seeds[s];
                seed.x = (seed.x- bbox_lower.x) / (bbox_upper.x - bbox_lower.x) * (maxval - minval) +  minval;
                seed.y = (seed.y- bbox_lower.y) / (bbox_upper.y - bbox_lower.y) * (maxval - minval) +  minval;
                for(int i = 0; i < file_cycles.size(); i++){
                    data.push_back(std::make_tuple(seed, file_cycles[i]));
                }
            }
            
        };
        // Override the get method to load custom data.
        torch::data::Example<> get(size_t index) override {

            vec2f start = std::get<0>(data[index]);
            float t = std::get<1>(data[index]);
            torch::Tensor start_tensor = torch::tensor({start.x, start.y});
            auto start_tensor_reshape = start_tensor.view({1, 2});
            torch::Tensor t_tensor = torch::tensor({t});
            auto t_tensor_reshape = t_tensor.view({1, 1});

            return {start_tensor, t_tensor};
        };

        // Override the size method to infer the size of the data set.
        torch::optional<size_t> size() const override {
            return data.size();
        };

};

class CustomDataset3D: public torch::data::Dataset<CustomDataset3D>
{
    private:
        std::vector<std::tuple<vec3f, float>> data;
    public:
        explicit CustomDataset3D(std::vector<vec3f> seeds, std::vector<float> file_cycles, vec3f bbox_lower, vec3f bbox_upper)
        {
            float minval = -1.f;
            float maxval = 1.f;
            for(int s = 0; s < seeds.size(); ++s){
                vec3f seed = seeds[s];
                seed.x = (seed.x- bbox_lower.x) / (bbox_upper.x - bbox_lower.x) * (maxval - minval) +  minval;
                seed.y = (seed.y- bbox_lower.y) / (bbox_upper.y - bbox_lower.y) * (maxval - minval) +  minval;
                seed.z = (seed.z- bbox_lower.z) / (bbox_upper.z - bbox_lower.z) * (maxval - minval) +  minval;
                for(int i = 0; i < file_cycles.size(); i++){
                    data.push_back(std::make_tuple(seed, file_cycles[i]));
                }
            }
            
        };
        // Override the get method to load custom data.
        torch::data::Example<> get(size_t index) override {

            vec3f start = std::get<0>(data[index]);
            float t = std::get<1>(data[index]);
            torch::Tensor start_tensor = torch::tensor({start.x, start.y, start.z});
            auto start_tensor_reshape = start_tensor.view({1, 3});
            torch::Tensor t_tensor = torch::tensor({t});
            auto t_tensor_reshape = t_tensor.view({1, 1});

            return {start_tensor, t_tensor};
        };

        // Override the size method to infer the size of the data set.
        torch::optional<size_t> size() const override {
            return data.size();
        };

};