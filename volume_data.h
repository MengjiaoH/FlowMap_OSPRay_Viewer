#pragma once

#include <memory>
#include <vector>
#include <ospray/ospray.h>
#include <ospray/ospray_cpp.h>
#include <ospray/ospray_cpp/ext/rkcommon.h>
#include <rkcommon/math/box.h>
#include <rkcommon/math/vec.h>

using namespace ospray;
using namespace rkcommon;

struct VolumeBrick {
    cpp::Volume brick;
    cpp::VolumetricModel model;
    math::box3f bounds;
    math::vec3i dims;
    std::shared_ptr<std::vector<uint8_t>> voxel_data;
    std::vector<float> ftle;

    math::vec2f value_range;
};

struct Trajectory {
    cpp::Geometry curve_geo;
    cpp::GeometricModel model;
    std::vector<math::vec4f> position_radius;
    std::vector<math::vec4f> colors;
    std::vector<uint> indices; 
    
};