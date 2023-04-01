
add_library(utils   place_seeds.cpp 
                    writer.cpp
					sobol.cpp)

target_link_libraries(utils PUBLIC rkcommon::rkcommon)
set_target_properties(utils PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON)