#include "place_seeds.h"

std::vector<vec2f> place_sobol_seeds_2d(vec2f x_range, vec2f y_range, int num_seeds, float z_start)
{
    std::vector<vec2f> seeds(num_seeds, vec2f(0));
    // srand((unsigned) time(NULL));
    char* file;
    double **P = sobol_points(num_seeds, 2, file); 
    for(int i = 0; i < num_seeds; i++){
        float x = (x_range.y - x_range.x) * P[i][0] + x_range.x;
        float y = (y_range.y - y_range.x) * P[i][1] + y_range.x;
        seeds[i].x = x;
        seeds[i].y = y;
    }
    return seeds;
}

std::vector<vec3f> place_sobol_seeds_3d(vec2f x_range, vec2f y_range, vec2f z_range, int num_seeds)
{
    std::vector<vec3f> seeds(num_seeds, vec3f(0, 0, 0));

    // float *temp = i4_sobol_generate(3, num_seeds, 0);
    char file[] = "new-joe-kuo-6.21201";
    double **P = sobol_points(num_seeds, 3, file); 
    for(int i = 0; i < num_seeds; i++){
        // float x = temp[2 * i];
        float x = (float)P[i][0];
        x = (x_range.y - x_range.x) * x + x_range.x;
        // float y = temp[2 * i + 1];
        float y = (float)P[i][1];
        y = (y_range.y - y_range.x) * y + y_range.x;
        // float z = temp[2 * i + 2];
        float z = (float)P[i][2];
        z = (z_range.y - z_range.x) * z + z_range.x;
        seeds[i].x = x;
        seeds[i].y = y;
        seeds[i].z = z;
    }
    return seeds;
}

std::vector<vec3f> place_random_seeds_2d(vec2f x_range, vec2f y_range, int num_seeds, float z_start)
{
    std::vector<vec3f> seeds(num_seeds, vec3f(0, 0, 0));
    srand((unsigned) time(NULL));
    for(int i = 0; i < num_seeds; i++){
        float x = (x_range.y - x_range.x) * ((float) rand() / RAND_MAX) + x_range.x;
        float y = (y_range.y - y_range.x) * ((float) rand() / RAND_MAX) + y_range.x;
        // float z = (z_range.y - z_range.x) * ((float) rand() / RAND_MAX) + z_range.x;
        seeds[i].x = x;
        seeds[i].y = y;
        seeds[i].z = z_start;
        // std::cout << x << " " << y << "\n";
    }
    return seeds;
}


std::vector<vec3f> place_random_seeds_3d(vec2f x_range, vec2f y_range, vec2f z_range, int num_seeds)
{
    std::vector<vec3f> seeds(num_seeds, vec3f(0));
    srand((unsigned) time(NULL));
    for(int i = 0; i < num_seeds; i++){
        float x = (x_range.y - x_range.x) * ((float) rand() / RAND_MAX) + x_range.x;
        float y = (y_range.y - y_range.x) * ((float) rand() / RAND_MAX) + y_range.x;
        float z = (z_range.y - z_range.x) * ((float) rand() / RAND_MAX) + z_range.x;
        seeds[i].x = x;
        seeds[i].y = y;
        seeds[i].z = z;
        // std::cout << x << " " << y << "\n";
    }
    return seeds;
}

std::vector<vec3f> place_uniform_seeds_2d(vec2f x_range, vec2f y_range, vec2i dims, float z_start)
{
    
    int num_seeds = dims.x * dims.y;
    std::vector<vec3f> seeds(num_seeds, vec3f(0,0,0));
    float x_interval = (x_range.y - x_range.x) / (dims.x - 1);
    float y_interval = (y_range.y - y_range.x) / (dims.y - 1);
    for(int j = 0; j < dims.y; j++)
	{
		for(int i = 0; i < dims.x; i++)
		{
			int index = dims.x * j + i;
			float x = i * x_interval + x_range.x;
			float y = j * y_interval + y_range.x;
            vec3f p = vec3f(x, y, z_start);
            seeds[index] = p;
		}
	}
    return seeds;
}

std::vector<vec3f> place_uniform_seeds_3d(vec2f x_range, vec2f y_range, vec2f z_range, vec3i dims)
{
    
    int num_seeds = dims.x * dims.y * dims.z;
    std::vector<vec3f> seeds(num_seeds, vec3f(0));
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
                vec3f p = vec3f(x, y, z);
                seeds[index] = p;
            }
	    }
    }
    return seeds;
} 

std::vector<vec3f> load_seeds_txt(std::string seeds_file, int num_seeds, bool is_z, float z_values)
{
    std::vector<vec3f> seeds(num_seeds, vec3f(0,0,0));
    std::fstream newfile;
    std::vector<float> temp;
    newfile.open(seeds_file.c_str(), std::ios::in);
    if (newfile.is_open()){   //checking whether the file is open
      std::string tp;
      while(getline(newfile, tp)){ //read data from file object and put it into string.
        std::stringstream ss(tp);
        float value;
        while(ss >> value){
            temp.push_back(value);
        } 
      }
      newfile.close(); //close the file object.
   }
   for(int i = 0 ; i < num_seeds; i++){
       if (is_z){
            vec3f p = vec3f(temp[2*i], temp[2*i + 1], z_values);
            seeds[i] = p;
       }else{
            vec3f p = vec3f(temp[3*i], temp[3*i + 1], temp[3*i + 2]);
            seeds[i] = p;
       }
       
   }
   return seeds;
}
