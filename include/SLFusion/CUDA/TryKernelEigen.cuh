#ifndef __SLFUSION_CUDA_KERNELEIGEN_CUH__
#define __SLFUSION_CUDA_KERNELEIGEN_CUH__

#include "SLFusion/CUDA/CUDACommon.cuh"

namespace slf_cuda
{

int crExponent(const CRReal_t* input, int rows, int cols, int cStep, CRReal_t* output);  

}

#endif // __SLFUSION_CUDA_KERNELEIGEN_CUH__