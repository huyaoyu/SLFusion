#ifndef __SLFUSION_CUDA_CUDATESTS_CUH__
#define __SLFUSION_CUDA_CUDATESTS_CUH__

#include "TopCommon.hpp"
#include "CUDACommon.cuh"

namespace slf_cuda
{

/**
 * \param out Each pixel location has the same channel number.
 */
int test_read_window_from_ocv(int H, int W, int channels, const Intensity_t* img, 
    int kernelSize, int numKernels, CRReal_t* out);

}

#endif /* __SLFUSION_CUDA_CUDATESTS_CUH__ */