cmake_minimum_required( VERSION 3.4 )
project ( CABAC )

set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3" )

set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

############### dependencies ######################

include_directories( 
 ${PROJECT_SOURCE_DIR}
)


add_library( codec SHARED
    	CABAC_ArithmeticDecoder.cpp
   	CABAC_ArithmeticEncoder.cpp
    	CABAC_BitstreamFile.cpp
    	ContextModel.cpp
)

add_executable( cabac CABAC.cpp )
target_link_libraries(	cabac
	codec
)









