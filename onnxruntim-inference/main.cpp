#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <filesystem>
#include <tbb/parallel_for.h>
#include "onnxruntime_cxx_api.h"
#include "place_seeds.h"
#include "writer.h"

using namespace std::chrono;

// int enable_cuda(OrtSessionOptions* session_options) {
//   // OrtCUDAProviderOptions is a C struct. C programming language doesn't have constructors/destructors.
//   OrtCUDAProviderOptions o;
//   // Here we use memset to initialize every field of the above data struct to zero.
//   memset(&o, 0, sizeof(o));
//   // But is zero a valid value for every variable? Not quite. It is not guaranteed. In the other words: does every enum
//   // type contain zero? The following line can be omitted because EXHAUSTIVE is mapped to zero in onnxruntime_c_api.h.
//   o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
//   o.gpu_mem_limit = SIZE_MAX;
//   OrtStatus* onnx_status = g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options, &o);
//   if (onnx_status != NULL) {
//     const char* msg = g_ort->GetErrorMessage(onnx_status);
//     fprintf(stderr, "%s\n", msg);
//     g_ort->ReleaseStatus(onnx_status);
//     return -1;
//   }
//   return 0;
// }

int main(int argc, char** argv)
{
    int dim = 2;
    // int start_fm = 0;
    // int stop_fm = 40;
    int num_fm = 100;
    int num_models = 1;
    float minval = -1.f;
    float maxval = 1.f;
    int interval = 1;
    float step_size = 0.15f;
    float t_start = 1.f * interval * step_size;
    float t_end = num_fm * interval * step_size;
    std::string outfile = "/home/mengjiao/Desktop/Examples/ONNX_Inference_C++/results/Hurricane/traj_";

    // vec3f bbox_lower = vec3f(-1.914290000000000089e-02, 9.905020000000000380e+01, -8.307750000000000412e-01);
    // vec3f bbox_upper = vec3f(1.770010000000000048e+02, -2.660919999999999952e-02, 9.904529999999999745e+01);
    // ABC
    // vec3f bbox_lower = vec3f(-9.426629999999999399e+00, -6.988199999999999967e+00, -8.096339999999999648e+00);
    // vec3f bbox_upper = vec3f(1.572780000000000022e+01, 1.326750000000000007e+01, 1.440070000000000050e+01);
    //Hurricane
    vec3f bbox_lower0 = vec3f(168.4770050048828, 121.2020034790039, -0.007076070178300142);
    vec3f bbox_upper0 = vec3f(500.0580139160156, 436.82501220703125, 99.25189971923828);

    vec3f bbox_lower1 = vec3f(-1.047129999999999939e+01, -8.274760000000000559e+00, -6.488919999999999411e-02);
    vec3f bbox_upper1 = vec3f(5.113299999999999841e+02, 5.063500000000000227e+02, 1.000139999999999958e+02);
    std::vector<vec3f> bbox_lower ={bbox_lower0, bbox_lower1};
    std::vector<vec3f> bbox_upper ={bbox_upper0, bbox_upper1};

    float offset = 0.00;
    // vec2f x_range = vec2f(0 + offset, 99 - offset);
    // vec2f y_range = vec2f(0 + offset, 177 - offset);
    // vec2f z_range = vec2f(0 + offset, 99 - offset);

    //ABC
    // vec2f x_range = vec2f(0 + offset, 2 * M_PI - offset);
    // vec2f y_range = vec2f(0 + offset, 2 * M_PI - offset);
    // vec2f z_range = vec2f(0 + offset, 2 * M_PI - offset);

    // Hurricane 
    vec2f x_range = vec2f(249 + offset, 499 - offset);
    vec2f y_range = vec2f(150 + offset, 400 - offset);
    vec2f z_range = vec2f(0 + offset, 99 - offset);

    int num_seeds = std::stoi(argv[2]);

    std::vector<vec3f> seeds = place_sobol_seeds_3d(x_range, y_range, z_range, num_seeds);
    
    std::string instanceName{"flowmap-inference"};
    std::string modelFilepath = argv[1]; // .onnx file

    std::cout << num_seeds << " " << modelFilepath << "\n";

    std::vector<std::filesystem::path> modelFilepaths;
    // for (const auto& entry : std::filesystem::directory_iterator{modelFilepath}) {
    //     if (entry.is_regular_file() && (entry.path().extension() == ".onnx")) {
    //         std::cout << entry.path().filename() << "\n";
    //         modelFilepaths.push_back(entry.path().filename());
    //     }
    // }
    // modelFilepaths.push_back(modelFilepath);
    
    std::vector<const char*> input_names = {"input_1", "input_2"};
    std::vector<const char*> output_names = {"output1"};
    std::vector<int64_t> input_dims1 = {num_seeds * num_fm, 3};
    std::vector<int64_t> input_dims2 = {num_seeds * num_fm, 1};
    std::vector<int64_t> out_dims = {num_seeds * num_fm, 3};
    // std::vector<std::vector<vec3f>> trajs;
    std::vector<std::vector<vec3f>> trajs(num_seeds, std::vector<vec3f>(num_fm * 1, vec3f(0)));
    
    float loadingTime = 0.f;
    float preprocessTime = 0.f;
    float inferenceTime = 0.f;
    
    
    // tbb::parallel_for( tbb::blocked_range<int>(0, num_models),[&](tbb::blocked_range<int> r)
    // {
        // for (int m=r.begin(); m<r.end(); ++m){
        std::vector<float> seeds_flat(seeds.size() * 3 * num_fm, 0.f);
        std::vector<float> file_cycles(seeds.size() * num_fm, 0.f);
        
        for(int m = 0; m < num_models; m++){
            std::cout << "m: " << m << "\n";
            // Start Preprocess
            auto start_preprocess = high_resolution_clock::now();
            for(int s = 0; s < seeds.size(); ++s){
                vec3f seed = seeds[s];
                float x = (seed.x- bbox_lower[m].x) / (bbox_upper[m].x - bbox_lower[m].x) * (maxval - minval) +  minval;
                float y = (seed.y- bbox_lower[m].y) / (bbox_upper[m].y - bbox_lower[m].y) * (maxval - minval) +  minval;
                float z = (seed.z- bbox_lower[m].z) / (bbox_upper[m].z - bbox_lower[m].z) * (maxval - minval) +  minval;
                    for(int i = 1; i < num_fm + 1; i++){
                        // data.push_back(std::make_tuple(seed, file_cycles[i]));
                        
                        seeds_flat[3 * (s * num_fm + (i - 1)) + 0] = x;
                        seeds_flat[3 * (s * num_fm + (i - 1)) + 1] = y;
                        seeds_flat[3 * (s * num_fm + (i - 1)) + 2] = z;
                        file_cycles[s * num_fm + i - 1] = (i * interval * step_size - t_start) / (t_end - t_start) * (maxval - minval) +  minval;
                }
            }
            auto stop_preprocess = high_resolution_clock::now();
            auto duration_preprocess = duration_cast<milliseconds>(stop_preprocess - start_preprocess);
            preprocessTime += duration_preprocess.count();

            // Create Session
            auto start = high_resolution_clock::now();

            Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
            Ort::SessionOptions sessionOptions;
            sessionOptions.SetIntraOpNumThreads(20);
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            sessionOptions.AppendExecutionProvider_CUDA(cuda_options);

            // std::string filepath = modelFilepath + modelFilepaths[m].string();
            std::string filepath = modelFilepath;
            // std::cout << "model file path: " << filepath << std::endl;
                        
            Ort::Session session = Ort::Session(env, filepath.c_str(), sessionOptions);
            
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop - start);
            loadingTime = duration.count() + loadingTime;
            // std::cout << "Loading model cost: " << duration.count() << " ms" << std::endl;
            
            
            auto start_inference = high_resolution_clock::now();

            Ort::AllocatorWithDefaultOptions allocator;
            std::vector<Ort::Value> inputTensors;
            Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
            
            inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, const_cast<float*>(seeds_flat.data()), seeds_flat.size(), input_dims1.data(), input_dims1.size()));
            inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, const_cast<float*>(file_cycles.data()), file_cycles.size(), input_dims2.data(), input_dims2.size()));
    
            std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                        inputTensors.data(), 2, output_names.data(), 1);
            auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
            
            assert(type_info.GetShape() == out_dims);
            size_t total_len = type_info.GetElementCount();
            float* results = ort_outputs[0].GetTensorMutableData<float>();
            for(int n = 0; n < num_seeds; n++){
                std::vector<vec3f> traj;
                int index_start = n * num_fm;
                int index_end = (n+1) * num_fm - 1;
                
                // std::cout << index_start << " " << index_end << std::endl;
                for(int s = index_start; s <= index_end; ++s){
                    vec3f pos = vec3f(results[3 * s], results[3 * s + 1], results[3 * s + 2]);
                    pos.x = (pos.x - minval) / (maxval - minval) * (bbox_upper[m].x - bbox_lower[m].x) + bbox_lower[m].x;
                    pos.y = (pos.y - minval) / (maxval - minval) * (bbox_upper[m].y - bbox_lower[m].y) + bbox_lower[m].y;
                    pos.z = (pos.z - minval) / (maxval - minval) * (bbox_upper[m].z - bbox_lower[m].z) + bbox_lower[m].z;
                    // traj[s - index_start + 1] = pos;
                    traj.push_back(pos);
                }
            // std::cout << "here " << n << " " << traj[num_fm -1] << "\n";
            // input_seeds[n] = vec3f(traj[num_fm-1].x, traj[num_fm-1].y, traj[num_fm-1].z);
            
                trajs[n] = traj;
                // trajs.push_back(traj);
            }   

            auto stop_inference = high_resolution_clock::now();
            auto duration_inference = duration_cast<milliseconds>(stop_inference - start_inference);
            inferenceTime += duration_inference.count();
            // std::cout << "Inference cost: " << duration_inference.count()  << " ms" << std::endl;


        }
    // });

                
             std::cout << "Loading cost: " << loadingTime << " ms" << std::endl;
             std::cout << "Preprocess cost: " << preprocessTime << " ms" << std::endl;
             std::cout << "Inference cost: " << inferenceTime << " ms" << std::endl;

    // for (size_t i = 0; i != total_len; ++i) {
    //     std::cout << i << " " << f[i] << "\n";
    // }
    // std::vector<std::vector<vec3f>> trajs(num_seeds, std::vector<vec3f>(num_fm * 1, vec3f(0)));
    
    // std::vector<vec3f> traj(num_fm, vec3f(0));
    
    
    

    // std::cout << "debug0" << "\n";
    

    // write_to_txt_vec3f(trajs, outfile);
    // std::cout << "debug1" << "\n";
    return 0;
}