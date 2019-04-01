#include "TopCommon.hpp"

#include <boost/filesystem.hpp>
#include <cmath>
#include <fstream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "SLFusion/BilateralWindowMatcher.hpp"
#include "SLFusion/SLCommon.hpp"

#include "SLFusion/CUDA/CUDAMatchImage.cuh"

using namespace cv;
using namespace Eigen;
using namespace slf;
using namespace slf_cuda;

void BilateralWindowMatcher::match_image(
    const Mat& refMat, const Mat& tstMat, 
    const Mat& refInt, const Mat& tstInt,
    const Mat& refMInt, const Mat& tstMInt,
    int minDisp, int maxDisp,
    Mat& disp )
{
    int res;

    CUDAMatcher cm = CUDAMatcher();

    res = cm.cuda_match_image( 
        refMat.rows, refMat.cols, refMat.channels(),
        refMat.ptr<Intensity_t>(), tstMat.ptr<Intensity_t>(), 
        refInt.ptr<Integral_t>(), tstInt.ptr<Integral_t>(), 
        refMInt.ptr<Integral_t>(), tstMInt.ptr<Integral_t>(), 
        mKernelSize, mNumKernels,
        mWss.data(),
        minDisp, maxDisp, 
        disp.ptr<CRReal_t>() );
}