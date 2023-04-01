// Frances Y. Kuo
//
// Email: <f.kuo@unsw.edu.au>
// School of Mathematics and Statistics
// University of New South Wales
// Sydney NSW 2052, Australia
// 
// Last updated: 21 October 2008
//
//   You may incorporate this source code into your own program 
//   provided that you
//   1) acknowledge the copyright owner in your program and publication
//   2) notify the copyright owner by email
//   3) offer feedback regarding your experience with different direction numbers
//
//
// -----------------------------------------------------------------------------
// Licence pertaining to sobol.cc and the accompanying sets of direction numbers
// -----------------------------------------------------------------------------
// Copyright (c) 2008, Frances Y. Kuo and Stephen Joe
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
// 
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
// 
//     * Neither the names of the copyright holders nor the names of the
//       University of New South Wales and the University of Waikato
//       and its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// -----------------------------------------------------------------------------

#include <cstdlib> // *** Thanks to Leonhard Gruenschloss and Mike Giles   ***
#include <cmath>   // *** for pointing out the change in new g++ compilers ***

#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

// ----- SOBOL POINTS GENERATOR BASED ON GRAYCODE ORDER -----------------
// INPUT: 
//   N         number of points  (cannot be greater than 2^32)
//   D         dimension  (make sure that the data file contains enough data!!)      
//   dir_file  the input file containing direction numbers
//
// OUTPUT:
//   A 2-dimensional array POINTS, where
//     
//     POINTS[i][j] = the jth component of the ith point,
//   
//   with i indexed from 0 to N-1 and j indexed from 0 to D-1
//
// ----------------------------------------------------------------------

double **sobol_points(unsigned N, unsigned D, char *dir_file);