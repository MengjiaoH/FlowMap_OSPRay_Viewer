// #pragma once
// #include "rkcommon/math/vec.h"
// #include <iostream>
// #include <string>
// #include <vector>
// #include <cmath>
// #include <sstream>
// #include <filesystem>
// #include <chrono>
// using namespace std::chrono;
// #include <vtkFloatArray.h>
// #include <vtkDoubleArray.h>
// #include <vtkSmartPointer.h>
// #include <vtkDataArray.h>
// #include <vtkPointData.h>
// #include <vtkPoints.h>
// #include <vtkCellData.h>
// #include <vtkAbstractArray.h>
// #include <vtkRectilinearGridReader.h>
// #include <vtkRectilinearGrid.h> 
// #include <vtkCellDataToPointData.h>
// #include <vtkDataSetWriter.h>
// #include <vtkStructuredPointsReader.h>
// #include <vtkStructuredPoints.h>
// #include <vtkXMLStructuredGridReader.h>
// #include <vtkStructuredGridReader.h>
// #include <vtkStructuredGrid.h>
// #include <vtkXMLImageDataReader.h>
// #include <vtkImageData.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <vtkCellLocator.h>
// #include <vtkm/Matrix.h>
// #include <vtkm/Types.h>
// #include <vtkm/cont/DataSet.h>
// #include <vtkm/cont/DataSetBuilderUniform.h>
// #include <vtkm/cont/DataSetBuilderExplicit.h>
// #include <vtkm/io/writer/VTKDataSetWriter.h>
// #include <vtkm/worklet/DispatcherMapField.h>
// #include <vtkm/worklet/WorkletMapField.h>
// #include <vtkm/cont/DynamicCellSet.h>
// #include <vtkm/cont/ArrayCopy.h>
// #include <vtkm/cont/ArrayHandle.h>
// #include <vtkm/cont/ArrayHandleIndex.h>
// #include <vtkm/cont/ArrayPortalToIterators.h>
// #include <vtkm/cont/DeviceAdapter.h>
// #include <vtkm/cont/ErrorFilterExecution.h>
// #include <vtkm/worklet/ParticleAdvection.h>
// #include <vtkm/worklet/particleadvection/GridEvaluators.h>
// // #include <vtkm/worklet/particleadvection/Integrators.h>
// #include <vtkm/worklet/particleadvection/Particles.h>
// #include <vtkm/worklet/particleadvection/RK4Integrator.h>
// #include <vtkm/cont/VariantArrayHandle.h>
// #include <vtkm/cont/ArrayHandleVirtual.h>
// #include <vtkm/worklet/lcs/GridMetaData.h>
// #include <vtkm/worklet/lcs/LagrangianStructureHelpers.h>
// #include <vtkm/worklet/LagrangianStructures.h>
// #include <vtkm/worklet/particleadvection/Stepper.h>

// #include "volume_data.h"

#include "load_vtk.h"

using namespace rkcommon::math;

VolumeBrick load_vtk_volume(const std::string filename, const vec3i dims)
{
    VolumeBrick brick;
   
    const math::vec3f grid_spacing = math::vec3f(1.f);
    brick.dims = dims;
    brick.bounds = math::box3f(math::vec3f(0), brick.dims * grid_spacing);

    brick.brick = cpp::Volume("structuredRegular");
    brick.brick.setParam("dimensions", brick.dims);
    brick.brick.setParam("gridSpacing", grid_spacing);
    size_t voxel_size = 4;
    brick.brick.setParam("voxelType", int(OSP_FLOAT));

    const size_t n_voxels = brick.dims.long_product();
    brick.voxel_data = std::make_shared<std::vector<uint8_t>>(n_voxels * voxel_size, 0);
    

    std::vector<float> voxel_data = std::vector<float>(n_voxels, 0.f);
    vtkSmartPointer<vtkStructuredPointsReader> reader = vtkSmartPointer<vtkStructuredPointsReader>::New();
    std::cout << "fname: " << filename << std::endl;
    reader->SetFileName(filename.c_str());
    reader->Update();	
    vtkSmartPointer<vtkStructuredPoints> mesh = vtkSmartPointer<vtkStructuredPoints>::New();

    
    mesh = reader->GetOutput();
    int num_pts = mesh->GetNumberOfPoints();
    vtkAbstractArray* a1 = mesh->GetPointData()->GetArray("ftle"); 
    vtkFloatArray* att1 = vtkFloatArray::SafeDownCast(a1);
    for( int i = 0; i < num_pts; i++){
        float f = att1 ->GetTuple1(i);
        voxel_data[i] = f;
    }
    brick.ftle = voxel_data;
    std::memcpy(brick.voxel_data->data(), &(*voxel_data.begin()), brick.voxel_data->size());
    float minval = *std::min_element(voxel_data.begin(), voxel_data.end());
    float maxval = *std::max_element(voxel_data.begin(), voxel_data.end());
    std::cout << "volume range: " << minval << " " << maxval << std::endl;
    brick.value_range = math::vec2f(minval, maxval);

    cpp::SharedData osp_data;
  
    osp_data = cpp::SharedData(reinterpret_cast<float *>(brick.voxel_data->data()),
                                   math::vec3ul(brick.dims));
   
    brick.brick.setParam("data", osp_data);
    brick.brick.commit();
    brick.model = cpp::VolumetricModel(brick.brick);

    return brick;

}

std::vector<math::vec3f> place_uniform_seeds_3d(math::vec2f x_range, math::vec2f y_range, math::vec2f z_range, math::vec3i dims)
{
    int num_seeds = dims.x * dims.y * dims.z;
    std::vector<math::vec3f> seeds(num_seeds, math::vec3f(0));
    float x_interval = (x_range.y - x_range.x) / (dims.x - 1);
    float y_interval = (y_range.y - y_range.x) / (dims.y - 1);
    float z_interval = (z_range.y - z_range.x) / (dims.z - 1);
    for (int k = 0; k < dims.z; k++){

        for(int j = 0; j < dims.y; j++)
	    {
            for(int i = 0; i < dims.x; i++)
            {
                int index = dims.x * dims.y * k + dims.x * j + i;
                float x = i * x_interval + x_range.x;
                float y = j * y_interval + y_range.x;
                float z = k * z_interval + z_range.x;
                vec3f p = math::vec3f(x, y, z);
                seeds[index] = p;
            }
	    }
    }
    return seeds;
} 


// std::vector<math::vec3f> read_vec3_from_txt(std::string filename)
// {
//     std::ifstream in(filename.c_str());
//     // Check if object is valid
//     if(!in){
//         std::cerr << "Cannot open the File : "<<filename<<std::endl;
//     }
//     std::string line;
//     std::vector<float> points;
//     while (std::getline(in, line)){
//         std::istringstream ss(line);
//         float num;
//         while (ss >> num){
//             points.push_back(num);
//         }
//     }
//     std::vector<vec3f> end;
//     for(int i = 0; i < points.size() / 3; i++){
//         vec3f v = vec3f(points[3 * i], points[3*i+1], points[3*i+2]);
//         end.push_back(v);
//     }
//     return end;
// }

math::vec3f load_txt_traj(const std::string filename, const int index, Trajectory &trajectories)
{
    std::ifstream in(filename.c_str());
    // Check if object is valid
    if(!in){
        std::cerr << "Cannot open the File : "<<filename<<std::endl;
    }
    std::string line;
    std::vector<float> points;
    while (std::getline(in, line)){
        std::istringstream ss(line);
        float num;
        while (ss >> num){
            points.push_back(num);
        }
    }
    int num_points = points.size() / 3;
    math::vec3f seed;
    math::vec3f last;

    for(int i = 0; i < num_points; i++){
        if (i == 0){
            seed = math::vec3f(points[3 * i], points[3*i+1], points[3*i+2]);
            last = math::vec3f(points[3 * (num_points-1)], points[3 * (num_points-1) + 1], points[3 * (num_points-1)+ 2]);
        }
        float dist = sqrt((last.x - seed.x) * (last.x - seed.x)  + (last.y - seed.y) * (last.y - seed.y) + (last.z - seed.z) * (last.z - seed.z)); 
        // std::cout << "dist " << dist << " " << seed << " " << last << "\n";
        if (seed.y < 140 && seed.x > 30 && seed.x < 70 && seed.z > 10 and seed.z < 90){
            trajectories.position_radius.push_back(math::vec4f(points[3 * i], points[3*i+1], points[3*i+2], 0.4f));
            // trajectories.colors.push_back(math::vec4f(245.f/255.f, 144.f/255.f, 66.f/255.f, 0.8f));
            trajectories.colors.push_back(math::vec4f(189.f/255.f, 189.f/255.f, 189.f/255.f, 1.0f));
    
            if (i < num_points - 1 ){
                trajectories.indices.push_back((index * num_points) + i);
            }
        }
        
    }

    return seed;

}

void interpolate_ftle(const std::vector<float> ftle, const std::vector<math::box4i> cell_vertices, std::vector<float> &scalars)
{
    for(int i = 0; i < cell_vertices.size(); i++){
        math::vec4i lower = cell_vertices[i].lower;
        math::vec4i upper = cell_vertices[i].upper;
        // std::cout << "lower: " << lower << "\n";
        // std::cout << "upper: " << upper << "\n";
        // std::cout << ftle[upper.x] << "\n";
        float f0 = (ftle[lower.x] + ftle[lower.y] + ftle[lower.z] + ftle[lower.w]) / 4.f;   
        float f1 = (ftle[upper.x] + ftle[upper.y] + ftle[upper.z] + ftle[upper.w]) / 4.f;   
        // std::cout << "value: " << (f0 + f1) / 2.f << "\n";
        scalars.push_back((f0 + f1) / 2.f);            
    }
}

std::vector<std::vector<double>> load_scalar_for_interpolation(const std::string path_prefix, const std::vector<std::filesystem::path> filenames, const std::vector<math::box4i> cell_vertices, std::vector<math::vec2d> &scalar_ranges)
{
    int num_steps = filenames.size();
    std::vector<std::vector<double>> res(filenames.size(), std::vector<double>(cell_vertices.size(), 0.f));
    for(int f = 0; f < num_steps; f++){
        std::vector<double> temp = std::vector<double>(cell_vertices.size(), 0.f);
        std::string cur =  path_prefix + filenames[f].string();
        vtkSmartPointer<vtkStructuredPointsReader> reader = vtkSmartPointer<vtkStructuredPointsReader>::New();
        // std::cout << "fname: " << cur << std::endl;
        reader->SetFileName(cur.c_str());
        reader->Update();	
        vtkSmartPointer<vtkStructuredPoints> mesh = vtkSmartPointer<vtkStructuredPoints>::New();
        mesh = reader->GetOutput();
        int num_pts = mesh->GetNumberOfPoints();
        // std::cout << "num pts: " << num_pts << "\n";
        vtkAbstractArray* a1 = mesh->GetPointData()->GetArray("Result"); 
        vtkDoubleArray* att1 = vtkDoubleArray::SafeDownCast(a1);
        std::vector<double> mag_array(num_pts, 0);
        for(int i = 0; i < num_pts; ++i){
            mag_array[i] = att1 ->GetTuple1(i);
        }
        
        double minval = *std::min_element(mag_array.begin(), mag_array.end());
        double maxval = *std::max_element(mag_array.begin(), mag_array.end());
        scalar_ranges.push_back(math::vec2d(minval, maxval));
        for(int i = 0; i < cell_vertices.size(); i++){
            math::vec4i lower = cell_vertices[i].lower;
            math::vec4i upper = cell_vertices[i].upper;
            // std::cout << "lower: " << lower << "\n";
            // std::cout << "upper: " << upper << "\n";
            // std::cout << mag_array[lower.x] << "\n";
            double f0 = (mag_array[lower.x] + mag_array[lower.y] + mag_array[lower.z] + mag_array[lower.w]) / 4.f;   
            double f1 = (mag_array[upper.x] + mag_array[upper.y] + mag_array[upper.z] + mag_array[upper.w]) / 4.f;   
            // std::cout << "value: " << (f0 + f1) / 2.f << "\n";
            temp[i] = (f0 + f1) / 2.f;            
        }
        res[f] = temp;
    }
    return res;
}