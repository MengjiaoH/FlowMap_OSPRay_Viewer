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
    std::string outfile = "/home/mengjiao/Desktop/FlowMap-Viewer/CPP_Inference/results/gerris_2400/traj_";
    // load model 
    if (argc != 2) {
        std::cerr << "usage: viewer <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        // kCUDA
        module = torch::jit::load(argv[1], torch::kCPU);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::cout << "load ok\n";

    int dim = 2;
    int start_fm = 0;
    int stop_fm = 60;
    vec2f bbox_lower = vec2f(0, 0);
    vec2f bbox_upper = vec2f(1, 1);
    float minval = -1.f;
    float maxval = 1.f;
    int interval = 5;
    float step_size = 0.01f;
    float t_start = 0.f;
    float t_end = (stop_fm - start_fm) * interval * step_size;

    float offset = 0.05;
    vec2f x_range = vec2f(0.000976563 + offset, 0.999023 - offset);
    vec2f y_range = vec2f(0.000976563 + offset, 0.999023 - offset);
    int num_seeds = 10;
    float z = 0.f;

    // Place seeds 
    std::vector<vec2f> seeds = place_sobol_seeds_2d(x_range, y_range, num_seeds, z);
    // //Debug
    // for(int i = 0; i < seeds.size(); i++){
    //     std::cout << "seeds " << i << " : [" << seeds[i].x << ", " << seeds[i].y << ", " << seeds[i].z << "]" << "\n";
    // }
    std::vector<float> file_cycles;
    for(int f = start_fm + 1; f < stop_fm + 1; f++){
        float file_cycle = ((f - start_fm -1) * interval * step_size - t_start) / (t_end - t_start) * (maxval - minval) +  minval;
        file_cycles.push_back(file_cycle);
    }
    
    
    std::vector<std::vector<vec2f>> trajs(num_seeds, std::vector<vec2f>((stop_fm - start_fm), vec2f(0)));
    /*
    Inference Batch seeds and file cycles
    */
    int64_t batch_size = 2000;
    int64_t num_workers = 2;
    std::vector<vec2f> traj((stop_fm - start_fm + 1), vec2f(0));
    std::vector<float> results;
    auto start = high_resolution_clock::now();
        
    auto dataset = CustomDataset(seeds, file_cycles, bbox_lower, bbox_upper).map(torch::data::transforms::Stack<>());
    // auto data_loader = torch::data::make_data_loader(std::move(dataset),torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers).enforce_ordering(true));
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset), torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers));
    for (torch::data::Example<>& batch : *data_loader) {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(batch.data.to(torch::kCPU));
        inputs.push_back(batch.target.to(torch::kCPU));
        auto output = module.forward(inputs).toTensor();
        std::vector<float> float_traj(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
        results.insert(results.end(), float_traj.begin(), float_traj.end());
    }
    // std::cout << "results: " << results.size() << "\n";

    for(int n = 0; n < num_seeds; n++){
        std::vector<vec2f> traj((stop_fm - start_fm + 1), vec2f(0));
        traj[0] = seeds[n];
        int index_start = n * (stop_fm - start_fm);
        int index_end = (n+1) * (stop_fm - start_fm) - 1;
        // std::cout << index_start << " " << index_end << std::endl;
        for(int s = index_start; s <= index_end; ++s){
            vec2f pos = vec2f(results[2 * s], results[2 * s + 1]);
            pos.x = (pos.x - minval) / (maxval - minval) * (bbox_upper.x - bbox_lower.x) + bbox_lower.x;
            pos.y = (pos.y - minval) / (maxval - minval) * (bbox_upper.y - bbox_lower.y) + bbox_lower.y;
            traj[s - index_start + 1] = pos;
        }
        // std::cout << traj.size() << "\n";
        trajs[n] = traj;
    }

    for(int t = 0; t < trajs.size(); t++){
        std::vector<vec2f> traj = trajs[t];
        std::cout << "seed " << t << ": " << "\n";
        for(int s = 0; s < traj.size(); s++){
            std::cout << traj[s] << " ";
        }
        std::cout << "\n";
    }
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time: " << duration.count() << " microseconds" << std::endl;

    write_to_txt_vec2f(trajs, outfile);
    
    return 0;
}


// torch::Tensor seed_tensor = torch::ones({1, dim});
// seed_tensor[0][0] = seeds[s].x;
// seed_tensor[0][1] = seeds[s].y;
// seed_tensor[0][2] = seeds[s].z;
// /*
// Inference Batch file cycles
// */
// int64_t batch_size = 2000;
// int64_t num_workers = 1;
// auto start = high_resolution_clock::now();
// for(int i = 0; i < num_seeds; ++i){
//     std::vector<vec2f> traj((stop_fm - start_fm + 1), vec2f(0));
//     vec2f seed = seeds[i];
//     traj[0] = seed;
//     seed.x = (seed.x- bbox_lower.x) / (bbox_upper.x - bbox_lower.x) * (maxval - minval) +  minval;
//     seed.y = (seed.y- bbox_lower.y) / (bbox_upper.y - bbox_lower.y) * (maxval - minval) +  minval;
//     auto dataset = CustomDatasetBatchFileCycles(seed, file_cycles).map(torch::data::transforms::Stack<>());
//     auto data_loader = torch::data::make_data_loader(std::move(dataset),torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers));
//     for (torch::data::Example<>& batch : *data_loader) {
//         std::vector<torch::jit::IValue> inputs;
//         inputs.push_back(batch.data.to(torch::kCUDA));
//         inputs.push_back(batch.target.to(torch::kCUDA));
//         auto output = module.forward(inputs).toTensor().to(torch::kCPU);
//         // std::cout << output << std::endl;
//         std::vector<float> float_traj(output.data_ptr<float>(), output.data_ptr<float>() + output.numel());
//         for(int f = 0; f < float_traj.size()/2; f++){
//             traj[f] = vec2f(float_traj[2 * f], float_traj[2 * f + 1]);
//         }
//     }
//     trajs[i] = traj;
// }
// auto stop = high_resolution_clock::now();
// auto duration = duration_cast<microseconds>(stop - start);
// std::cout << "Time: " << duration.count() << " microseconds" << std::endl;
/*
// Inference 
auto start = high_resolution_clock::now();
tbb::parallel_for( tbb::blocked_range<int>(0, seeds.size()), [&](tbb::blocked_range<int> r)
{
    for (int s = r.begin(); s < r.end(); ++s)
    {
        // for(int s = 0; s < seeds.size(); s++){
        std::vector<vec2f> traj((stop_fm - start_fm + 1), vec2f(0));
        traj[0] = seeds[s];
        // convert seed to tensor
        float x = (seeds[s].x - bbox_lower.x) / (bbox_upper.x- bbox_lower.x) * (maxval - minval) +  minval;
        float y = (seeds[s].y - bbox_lower.y) / (bbox_upper.y- bbox_lower.y) * (maxval - minval) +  minval;
        float seed[] = {x , y};
        torch::Tensor seed_tensor = torch::from_blob(seed, {1, dim});
        for(int f = start_fm; f < stop_fm; f++){
            // convert fm to tensor
            float t[] = {((f - start_fm -1) * interval * step_size - t_start) / (t_end - t_start) * (maxval - minval) +  minval};
            torch::Tensor time_tensor = torch::from_blob(t, {1, 1});
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(seed_tensor.to(torch::kCPU));
            inputs.push_back(time_tensor.to(torch::kCPU));
            auto output = module.forward(inputs).toTensor();
            // torch::Tensor out1 = output->elements()[0].toTensor();
            
            // output = output.to(torch::kCPU);
            // std::cout << "out tensor " << output << std::endl;
            // std::cout << "out tensor size " << out1[0].sizes() << std::endl;
            vec2f v = vec2f(*output.data_ptr<float>(), *output.data_ptr<float>() + output.numel());
            v.x = (v.x - minval) / (maxval - minval) * (bbox_upper.x - bbox_lower.x) + bbox_lower.x;
            v.y = (v.y - minval) / (maxval - minval) * (bbox_upper.y - bbox_lower.y) + bbox_lower.y;
            traj[f + 1] = v;
        }
        trajs[s] = traj;
    }
});
auto stop = high_resolution_clock::now();
auto duration = duration_cast<microseconds>(stop - start);
*/
// To get the value of duration use the count()
// member function on the duration object
// std::cout << "Time: " << duration.count() << " microseconds" << std::endl;
