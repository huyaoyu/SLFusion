#include "TopCommon.hpp"
#include "SLFusion/CUDA/CUDAMatchImage.cuh"

using namespace slf_cuda;

int CUDAMatcher::cuda_match_image( 
    int H, int W, int channels,
    const Intensity_t* refMat, const Intensity_t* tstMat, 
    const Integral_t* refInt,  const Integral_t* tstInt, 
    const Integral_t* refMInt, const Integral_t* tstMInt, 
    int kernelSize, int numKernels,
    const CRReal_t* ws,
    int minDisp, int maxDisp, 
    CRReal_t* disp)
{
    return 0;
}


