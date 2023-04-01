#pragma once 
#include <iostream>
#include <vector>
#include "rkcommon/math/vec.h"
using namespace rkcommon::math;

void crossProduct(vec3f &ans, vec3f &v1, vec3f &v2);

double dotProduct( vec3f &v1, vec3f &v2 );

void bary_tet( vec4f &ans, vec3f &p, vec3f &a, vec3f &b, vec3f &c, vec3f &d );