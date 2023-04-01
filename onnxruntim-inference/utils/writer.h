#pragma once 
#include <iostream>
#include <fstream>
#include <vector>
#include "rkcommon/math/vec.h"
using namespace rkcommon::math;

void save_fm_as_raw_3d(std::vector<std::vector<vec3f>> fms, std::string outdir);

void save_fm_as_raw_2d(std::vector<std::vector<vec2f>> fms, std::string outdir);

void write_to_txt_2d(std::vector<float> data, std::string outfile);

void write_to_txt_3d(std::vector<float> data, std::string outfile);

void write_to_txt_float(std::vector<float> average_ftle, std::string outfile);

void write_to_txt_vec3f(std::vector<std::vector<vec3f>> fms, std::string outfile);

void write_to_txt_vec2f(std::vector<std::vector<vec2f>> fms, std::string outfile);

void write_seeds_2d(std::vector<vec2f> seeds, std::string outfile);
void write_seeds_3d(std::vector<vec3f> seeds, std::string outfile);

std::vector<vec3f> read_vec3_from_txt(std::string filename, bool z_zeros);

std::vector<vec3f> read_vec2_from_txt(std::string filename);