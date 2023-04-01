#pragma once
#include "rkcommon/math/vec.h"
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include <filesystem>
#include <chrono>
using namespace std::chrono;
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkSmartPointer.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkCellData.h>
#include <vtkAbstractArray.h>
#include <vtkRectilinearGridReader.h>
#include <vtkRectilinearGrid.h> 
#include <vtkCellDataToPointData.h>
#include <vtkDataSetWriter.h>
#include <vtkStructuredPointsReader.h>
#include <vtkStructuredPoints.h>
#include <vtkXMLStructuredGridReader.h>
#include <vtkStructuredGridReader.h>
#include <vtkStructuredGrid.h>
#include <vtkXMLImageDataReader.h>
#include <vtkImageData.h>
#include <stdio.h>
#include <stdlib.h>
#include <vtkCellLocator.h>
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

#include "volume_data.h"
#include <ospray/ospray.h>
#include <ospray/ospray_cpp.h>

// using Vec3f = vtkm::Vec<vtkm::FloatDefault, 3>;


using namespace rkcommon::math;

VolumeBrick load_vtk_volume(const std::string filename, const vec3i dims);

// std::vector<math::vec3f> read_vec3_from_txt(std::string filename);

math::vec3f load_txt_traj(const std::string filename, const int index, Trajectory &trajectories);

std::vector<math::vec3f> place_uniform_seeds_3d(math::vec2f x_range, math::vec2f y_range, math::vec2f z_range, math::vec3i dims);

void interpolate_ftle(const std::vector<float> ftle, const std::vector<math::box4i> cell_vertices, std::vector<float> &scalars);

std::vector<std::vector<double>> load_scalar_for_interpolation(const std::string path_prefix, const std::vector<std::filesystem::path> filenames, const std::vector<math::box4i> cell_vertices, std::vector<math::vec2d> &scalar_ranges);