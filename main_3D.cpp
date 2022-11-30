#include <iostream>
#include <vector>
#include <chrono>
#include <torch/script.h> // One-stop header.
#include "rkcommon/math/vec.h"
#include "place_seeds.h"
#include <tbb/parallel_for.h>
#include "data_loader.h"
#include "writer.h"

using namespace rkcommon::math;
using namespace std::chrono;

int main(int argc, char **argv)
{
    std::string outfile = "/home/mengjiao/Desktop/FlowMap-Viewer/CPP_Inference/results/scalarflow/traj_";
    // load model 
    if (argc != 2) {
        std::cerr << "usage: viewer <path-to-exported-script-module>\n";
        return -1;
    }

    std::string prefix = argv[1];

    std::vector<torch::jit::script::Module> modules(2);
    auto start = high_resolution_clock::now();
    
    tbb::parallel_for( tbb::blocked_range<int>(0,2),[&](tbb::blocked_range<int> r)
    {
        for (int i=r.begin(); i<r.end(); ++i){
    // for(int i = 0; i < 2; i++){
        std::string model_dir = prefix + "model_" + std::to_string(i) + ".pt";
        try {
                // Deserialize the ScriptModule from a file using torch::jit::load().
                // kCUDA
                modules[i] = torch::jit::load(model_dir, torch::kCPU);
            }
            catch (const c10::Error& e) {
                std::cerr << "error loading the model\n";
                return -1;
            }
            std::cout << "load ok\n";
    }
    });

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time: " << duration.count() << " microseconds" << std::endl;

    
    
    int dim = 3;
    int start_fm = 0;
    int stop_fm = 20;
    int num_fm = 20;
    float minval = -1.f;
    float maxval = 1.f;
    int interval = 1;
    float step_size = 1.f;
    float t_start = 1.f;
    float t_end = num_fm * interval * step_size;

    vec3f bbox_lower = vec3f(-1.914290000000000089e-02, 9.905020000000000380e+01, -8.307750000000000412e-01);
    vec3f bbox_upper = vec3f(1.770010000000000048e+02, -2.660919999999999952e-02, 9.904529999999999745e+01);

    

    float offset = 0.00;
    vec2f x_range = vec2f(0 + offset, 99 - offset);
    vec2f y_range = vec2f(0 + offset, 177 - offset);
    vec2f z_range = vec2f(0 + offset, 99 - offset);
    int num_seeds = 100;

    // Place seeds 
    std::vector<vec3f> seeds = place_sobol_seeds_3d(x_range, y_range, z_range, num_seeds);
    // //Debug
    // for(int i = 0; i < seeds.size(); i++){
    //     std::cout << "seeds " << i << " : [" << seeds[i].x << ", " << seeds[i].y << ", " << seeds[i].z << "]" << "\n";
    // }


    std::vector<float> file_cycles;
    for(int f = 1; f < num_fm + 1; f++){
        float file_cycle = (f * interval * step_size - t_start) / (t_end - t_start) * (maxval - minval) +  minval;
        file_cycles.push_back(file_cycle);
    }
    
    
    std::vector<std::vector<vec3f>> trajs(num_seeds, std::vector<vec3f>(num_fm * modules.size(), vec3f(0)));
    
    
    // Inference Batch seeds and file cycles
    
    
    int64_t batch_size = 50;
    int64_t num_workers = 2;
    
    
    auto start_tracing = high_resolution_clock::now();

    std::vector<vec3f> input_seeds = seeds;
    
    tbb::parallel_for( tbb::blocked_range<int>(0,2),[&](tbb::blocked_range<int> r)
    {
        for (int i=r.begin(); i<r.end(); ++i){
    // for(int i = 0; i < modules.size(); i++){
        // std::vector<float> results(num_seeds * num_fm * dim, 0.f);
        std::vector<float> results;
        auto dataset = CustomDataset3D(input_seeds, file_cycles, bbox_lower, bbox_upper).map(torch::data::transforms::Stack<>());
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset), torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers));
        for (torch::data::Example<>& batch : *data_loader) {
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(batch.data.to(torch::kCPU));
            inputs.push_back(batch.target.to(torch::kCPU));
            auto output = modules[i].forward(inputs).toTensor();
            std::vector<float> float_traj(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
            results.insert(results.end(), float_traj.begin(), float_traj.end());
        }

        
        // for(int n = 0; n < num_seeds; n++){
        //     std::vector<vec3f> traj(num_fm, vec3f(0));
        //     int index_start = n * num_fm;
        //     int index_end = (n+1) * num_fm - 1;
        //     // std::cout << index_start << " " << index_end << std::endl;
        //     for(int s = index_start; s <= index_end; ++s){
        //         vec3f pos = vec3f(results[3 * s], results[3 * s + 1], results[3 * s + 2]);
        //         pos.x = (pos.x - minval) / (maxval - minval) * (bbox_upper.x - bbox_lower.x) + bbox_lower.x;
        //         pos.y = (pos.y - minval) / (maxval - minval) * (bbox_upper.y - bbox_lower.y) + bbox_lower.y;
        //         pos.z = (pos.z - minval) / (maxval - minval) * (bbox_upper.z - bbox_lower.z) + bbox_lower.z;
        //         traj[s - index_start + 1] = pos;
        //     }
        //     input_seeds[n] = traj[num_fm-1];
        // }
        
        
    }
    });
        
    // auto dataset = CustomDataset3D(seeds, file_cycles, bbox_lower, bbox_upper).map(torch::data::transforms::Stack<>());
    // // auto data_loader = torch::data::make_data_loader(std::move(dataset),torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers).enforce_ordering(true));
    // auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset), torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers));
    // for (torch::data::Example<>& batch : *data_loader) {
    //     std::vector<torch::jit::IValue> inputs;
    //     inputs.push_back(batch.data.to(torch::kCPU));
    //     inputs.push_back(batch.target.to(torch::kCPU));
    //     auto output = module.forward(inputs).toTensor();
    //     std::vector<float> float_traj(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
    //     results.insert(results.end(), float_traj.begin(), float_traj.end());
    // }
    // std::cout << "results: " << results.size() << "\n";

    // for(int n = 0; n < num_seeds; n++){
    //     std::vector<vec3f> traj((stop_fm - start_fm + 1), vec2f(0));
    //     traj[0] = seeds[n];
    //     int index_start = n * (stop_fm - start_fm);
    //     int index_end = (n+1) * (stop_fm - start_fm) - 1;
    //     // std::cout << index_start << " " << index_end << std::endl;
    //     for(int s = index_start; s <= index_end; ++s){
    //         vec3f pos = vec3f(results[3 * s], results[3 * s + 1], results[3 * s + 2]);
    //         pos.x = (pos.x - minval) / (maxval - minval) * (bbox_upper.x - bbox_lower.x) + bbox_lower.x;
    //         pos.y = (pos.y - minval) / (maxval - minval) * (bbox_upper.y - bbox_lower.y) + bbox_lower.y;
    //         pos.z = (pos.z - minval) / (maxval - minval) * (bbox_upper.z - bbox_lower.z) + bbox_lower.z;
    //         traj[s - index_start + 1] = pos;
    //     }
    //     // std::cout << traj.size() << "\n";
    //     trajs[n] = traj;
    // }

    // for(int t = 0; t < trajs.size(); t++){
    //     std::vector<vec3f> traj = trajs[t];
    //     std::cout << "seed " << t << ": " << "\n";
    //     for(int s = 0; s < traj.size(); s++){
    //         std::cout << traj[s] << " ";
    //     }
    //     std::cout << "\n";
    // }
    
    auto stop_tracing = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop_tracing - start_tracing);
    std::cout << "Time: " << duration.count() << " microseconds" << std::endl;

    // write_to_txt_vec3f(trajs, outfile);
    
    return 0;
}
