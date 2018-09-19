cmake_minimum_required(VERSION 2.8.3)
project(SLFusion)

## Compile as C++11
add_compile_options(-std=c++11)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

## Specify additional locations of header files
## Your package locations should be listed before other locations

SET( PROJECT_INCLUDES "../include" )
SET( OPENCV_ROOT "$ENV{OPENCV_ROOT}" )
SET( OPENCV_INCLUDES "${OPENCV_ROOT}/include" )
SET( OPENCV_LIBS "${OPENCV_ROOT}/lib" )

# Show the above variables.
MESSAGE(STATUS "OPENCV_ROOT = ${OPENCV_ROOT}")
MESSAGE(STATUS "OPENCV_LIBS = ${OPENCV_LIBS}")

include_directories(
# include
  ${PROJECT_INCLUDES}
  ${OPENCV_INCLUDES}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/stereo_xi_camera.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# >>>
# Search the ./src folder for .cpp files.

FILE(GLOB_RECURSE SRC_LIST *.cpp)

IF (SRC_LIST)
	FOREACH (SFILE ${SRC_LIST})
		MESSAGE(STATUS "CPP: ${SFILE}")
	ENDFOREACH (SFILE)
ENDIF (SRC_LIST) 

# Declare a C++ executable
# With catkin_make all packages are built within a single CMake context
# The recommended prefix ensures that target names across packages don't collide
add_executable(${PROJECT_NAME}_exe ${SRC_LIST})

## Add cmake target dependencies of the executable
## same as for the library above

FIND_LIBRARY( LIB_OPENCV_CORE NAMES opencv_core PATHS ${OPENCV_LIBS} NO_DEFAULT_PATH )
FIND_LIBRARY( LIB_OPENCV_HIGHGUI NAMES opencv_highgui PATHS ${OPENCV_LIBS} NO_DEFAULT_PATH )
FIND_LIBRARY( LIB_OPENCV_IMGPROC NAMES opencv_imgproc PATHS ${OPENCV_LIBS} NO_DEFAULT_PATH )
FIND_LIBRARY( LIB_OPENCV_IMGCODECS NAMES opencv_imgcodecs PATHS ${OPENCV_LIBS} NO_DEFAULT_PATH )
FIND_LIBRARY( LIB_BOOST_EXCEPTION NAMES boost_exception )

FIND_LIBRARY( ZLIB REQUIRED )

SET( CNPY_LIBRARIES "/usr/local/lib/libcnpy.so")

# Show the found libraries.
MESSAGE( STATUS "LIB_OPENCV_CORE = ${LIB_OPENCV_CORE}" )
MESSAGE( STATUS "LIB_OPENCV_HIGHGUI = ${LIB_OPENCV_HIGHGUI}" )
MESSAGE( STATUS "LIB_OPENCV_IMGPROC = ${LIB_OPENCV_IMGPROC}" )
MESSAGE( STATUS "LIB_OPENCV_IMGCODECS = ${LIB_OPENCV_IMGCODECS}" )
MESSAGE( STATUS "LIB_BOOST_EXCEPTION = ${LIB_BOOST_EXCEPTION}" )

# PCL
find_package(PCL 1.7 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

SET( CXX_FLAGS "-g" )
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${CXX_FLAGS}" )
SET( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

# Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_exe
  ${LIB_OPENCV_CORE} ${LIB_OPENCV_HIGHGUI} ${LIB_OPENCV_IMGPROC} ${LIB_OPENCV_IMGCODECS}
  ${LIB_BOOST_EXCEPTION}
  ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${CNPY_LIBRARIES} z
)

