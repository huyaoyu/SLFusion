#ifndef __SLFUSION_CUDA_CUDAMatchImage_CUH__
#define __SLFUSION_CUDA_CUDAMatchImage_CUH__

#include "TopCommon.hpp"
#include "CUDACommon.cuh"

namespace slf_cuda
{

class CUDAMatcher
{
public:
    CUDAMatcher() {}
    ~CUDAMatcher() {}

    int cuda_match_image( 
        int H, int W, int channels,
        const Intensity_t* refMat, const Intensity_t* tstMat, 
        const Integral_t* refInt,  const Integral_t* tstInt, 
        const Integral_t* refMInt, const Integral_t* tstMInt, 
        int kernelSize, int numKernels,
        const CRReal_t* wss,
        int minDisp, int maxDisp, 
        CRReal_t* disp);
};

}

#endif /* __SLFUSION_CUDA_CUDAMatchImage_CUH__ */