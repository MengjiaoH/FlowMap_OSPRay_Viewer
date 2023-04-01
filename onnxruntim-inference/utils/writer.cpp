// #include <iostream>
// #include <fstream>
// #include <vector>
// #include "rkcommon/math/vec.h"
// using namespace rkcommon::math;

#include "writer.h"

void save_fm_as_raw_3d(std::vector<std::vector<vec3f>> fms, std::string outdir)
{
    for(int i = 0; i < fms.size(); i++){
        
        std::string filename = outdir + "fm_" + std::to_string(i) + ".raw";
        std::ofstream wf(filename, std::ios::out | std::ios::binary);
        if(!wf) {
            std::cout << "Cannot open file!" << std::endl;
        }
        std::vector<float> end;
        for(int n = 0; n < fms[i].size(); n++){
            end.push_back(fms[i][n].x);
            end.push_back(fms[i][n].y);
            end.push_back(fms[i][n].z);
        }
        wf.write(reinterpret_cast<const char*>(&end[0]), end.size()*sizeof(float));
        wf.close();
    }
}

void save_fm_as_raw_2d(std::vector<std::vector<vec2f>> fms, std::string outdir)
{
    for(int i = 0; i < fms.size(); i++){
        
        std::string filename = outdir + "fm_" + std::to_string(i) + ".raw";
        std::ofstream wf(filename, std::ios::out | std::ios::binary);
        if(!wf) {
            std::cout << "Cannot open file!" << std::endl;
        }
        std::vector<float> end;
        for(int n = 0; n < fms[i].size(); n++){
            end.push_back(fms[i][n].x);
            end.push_back(fms[i][n].y);
        }
        wf.write(reinterpret_cast<const char*>(&end[0]), end.size()*sizeof(float));
        wf.close();
    }
}

void write_to_txt_2d(std::vector<float> data, std::string outfile)
{   
    std::ofstream wf(outfile.c_str(), std::ios::out);
    if(!wf) {
        std::cout << "Cannot open file!" << std::endl;
    }
    for(int j = 0; j < data.size(); j++){
        wf << data[2 * j] << " " << data[2 * j + 1] << " ";
    }
    wf << "\n";
    wf.close();
    if(!wf.good()) {
        std::cout << "Error occurred at writing time!" << std::endl;
    }
}

void write_to_txt_3d(std::vector<float> data, std::string outfile)
{   
    std::ofstream wf(outfile.c_str(), std::ios::out);
    if(!wf) {
        std::cout << "Cannot open file!" << std::endl;
    }
    for(int j = 0; j < data.size(); j++){
        wf << data[3 * j] << " " << data[3 * j + 1] << " " << data[3 * j + 2] << " ";
    }
    wf << "\n";
    wf.close();
    if(!wf.good()) {
        std::cout << "Error occurred at writing time!" << std::endl;
    }
}

void write_to_txt_float(std::vector<float> average_ftle, std::string outfile)
{   
    std::ofstream wf(outfile.c_str(), std::ios::out);
    if(!wf) {
        std::cout << "Cannot open file!" << std::endl;
    }
    for(int j = 0; j < average_ftle.size(); j++){
        wf << average_ftle[j] << " ";
    }
    wf << "\n";
    wf.close();
    if(!wf.good()) {
        std::cout << "Error occurred at writing time!" << std::endl;
    }
}

void write_to_txt_vec3f(std::vector<std::vector<vec3f>> fms, std::string outfile)
{
    for(int i = 0; i < fms.size(); ++i){
        std::string outfilename = outfile + std::to_string(i) + ".txt";
        // std::string outfile = "/home/mengjiao/Desktop/projects/Partical_Advection/result_data/Gerris/4600/fm_" + std::to_string(i) + ".txt";
        std::ofstream wf(outfilename.c_str(), std::ios::out);
        if(!wf) {
            std::cout << "Cannot open file!" << std::endl;
        }
        std::vector<vec3f> temp = fms[i];
        // wf.write((char *) &fms[i], sizeof(float) * fms[i].size() * 3);
        for(int j = 0; j < temp.size(); j++){
            wf << temp[j].x << " " << temp[j].y << " " << temp[j].z << "\n";
        }
        wf.close();
        if(!wf.good()) {
            std::cout << "Error occurred at writing time!" << std::endl;
        }
    }
}

void write_to_txt_vec2f(std::vector<std::vector<vec2f>> trajs, std::string outfile)
{
    for(int i = 0; i < trajs.size(); ++i){
        std::string outfilename = outfile + std::to_string(i) + ".txt";
        std::ofstream wf(outfilename.c_str(), std::ios::out);
        if(!wf) {
            std::cout << "Cannot open file!" << std::endl;
        }
        std::vector<vec2f> temp = trajs[i];
        // wf.write((char *) &fms[i], sizeof(float) * fms[i].size() * 3);
        for(int j = 0; j < temp.size(); j++){
            wf << temp[j].x << " " << temp[j].y << "\n";
        }
        wf.close();
        if(!wf.good()) {
            std::cout << "Error occurred at writing time!" << std::endl;
        }
    }
}

void write_seeds_2d(std::vector<vec2f> seeds, std::string outfile){
    std::ofstream wf(outfile.c_str(), std::ios::out);
    if(!wf) {
        std::cout << "Cannot open file!" << std::endl;
    }
    for(int j = 0; j < seeds.size(); j++){
        wf << seeds[j].x << " " << seeds[j].y << "\n ";
    }
    wf.close();
    if(!wf.good()) {
        std::cout << "Error occurred at writing time!" << std::endl;
    }
}

void write_seeds_3d(std::vector<vec3f> seeds, std::string outfile){
    std::ofstream wf(outfile.c_str(), std::ios::out);
    if(!wf) {
        std::cout << "Cannot open file!" << std::endl;
    }
    for(int j = 0; j < seeds.size(); j++){
        wf << seeds[j].x << " " << seeds[j].y << " " << seeds[j].z << "\n ";
    }
    wf.close();
    if(!wf.good()) {
        std::cout << "Error occurred at writing time!" << std::endl;
    }
}

std::vector<vec3f> read_vec3_from_txt(std::string filename, bool z_zeros)
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
    std::vector<vec3f> end;
    for(int i = 0; i < points.size() / 3; i++){
        if (z_zeros){
            vec3f v = vec3f(points[3 * i], points[3*i+1], 0.0f);
            end.push_back(v);
        }else{
            vec3f v = vec3f(points[3 * i], points[3*i+1], points[3*i+2]);
            end.push_back(v);
        }
        
    }
    return end;
}


std::vector<vec3f> read_vec2_from_txt(std::string filename)
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
    std::vector<vec3f> end;
    for(int i = 0; i < points.size() / 2; i++){
        vec3f v = vec3f(points[2 * i], points[2*i+1], 0.0f);
        end.push_back(v);
    }
    return end;
}