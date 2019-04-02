cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(SLF_CUDARoutines LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -g -G
)

set( PROJECT_INCLUDES "../../../../include" )

include_directories(
# include
  ${PROJECT_INCLUDES}
)

file(GLOB_RECURSE SRC_LIST *.cu)

if (SRC_LIST)
	foreach (SFILE ${SRC_LIST})
		message(STATUS "CUDA: ${SFILE}")
	endforeach (SFILE)
endif (SRC_LIST) 

add_library( slf_cr ${SRC_LIST} )

target_compile_features( slf_cr PUBLIC cxx_std_11 )
set_target_properties( slf_cr PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
