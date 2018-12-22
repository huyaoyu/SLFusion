#include <cmath>
#include <fstream>
#include <vector>

#include <opencv2/highgui.hpp>

#include "SLFusion/BilateralWindowMatcher.hpp"

using namespace cv;
using namespace Eigen;
using namespace slf;

typedef IMatrix_t IM_t;
typedef FMatrix_t FM_t;
typedef Real_t    R_t;

WeightColor::WeightColor()
{

}

WeightColor::WeightColor(int numKernels, IMatrix_t& knlIdxRow, IMatrix_t& knlIdxCol, Real_t gammaC)
: mNumKernels(numKernels),
  mKnlIdxRow(knlIdxRow), mKnlIdxCol(knlIdxCol),
  mGammaC(gammaC)
{
    mCenterIdx = (mNumKernels - 1) / 2;

    mVcMat  = Mat(mNumKernels, mNumKernels, CV_8UC1);
    mTvcMat = Mat(mNumKernels, mNumKernels, CV_8UC1);
}

WeightColor::~WeightColor()
{

}

WeightColor& WeightColor::operator=( const WeightColor& rhs )
{
    if ( &rhs == this )
    {
        return *this;
    }

    this->mNumKernels = rhs.mNumKernels;
    this->mCenterIdx  = rhs.mCenterIdx;
    this->mKnlIdxRow  = rhs.mKnlIdxRow;
    this->mKnlIdxCol  = rhs.mKnlIdxCol;
    this->mGammaC     = rhs.mGammaC;

    rhs.mVcMat.copyTo( this->mVcMat );
    rhs.mTvcMat.copyTo( this->mTvcMat );

    return *this;
}

template<typename _TN, typename _TD, typename _TT> 
static void mat_divide(const Mat& n, const Mat& d, Mat& dst)
{
    if ( n.size() != d.size() )
    {
        std::cout << "Size mismatch." << std::endl;

        std::stringstream ssN, ssD;

        ssN << "n.size() = (" << n.rows << ", " << n.cols << ").";
        ssD << "d.size() = (" << d.rows << ", " << d.cols << ").";

        EXCEPTION_DIMENSION_MISMATCH( n, ssN.str(), d, ssD.str() );
    }

    if ( n.size() != dst.size() )
    {
        std::cout << "Size mismatch." << std::endl;

        std::stringstream ssN, ssDst;

        ssN   << "n.size() = (" << n.rows << ", " << n.cols << ").";
        ssDst << "dst.size() = (" << d.rows << ", " << d.cols << ").";

        EXCEPTION_DIMENSION_MISMATCH( n, ssN.str(), dst, ssDst.str() );
    }

    if ( n.channels() != dst.channels() )
    {
        std::cout << "Channels mismatch." << std::endl;

        std::stringstream ssN, ssT;

        ssN << "n.channels() = "   << n.channels() << ".";
        ssT << "dst.channels() = " << dst.channels() << ".";

        EXCEPTION_DIMENSION_MISMATCH( n, ssN.str(), dst, ssT.str() );
    }

    if ( d.channels() != 1 )
    {
        std::cout << "Currently only support d.channels() = 1." << std::endl;
        EXCEPTION_BAD_ARGUMENT(d, "Currently only support d.channels() = 1."); 
    }

    const _TN* pN = NULL;
    const _TD* pD = NULL;
    _TT* pT = NULL;

    const int channels = n.channels();
    int posN = 0;

    for ( int i = 0; i < n.rows; ++i )
    {
        pN = n.ptr<_TN>(i);
        pD = d.ptr<_TD>(i);
        pT = dst.ptr<_TT>(i);

        posN = 0;

        for ( int j = 0; j < n.cols; ++j )
        {
            for ( int k = 0; k < channels; ++k )
            {
                *(pT + posN + k) = *(pN + posN + k) / *(pD + j);
            }

            posN += channels;
        }
    }
}

void WeightColor::put_average_color_values(
        InputArray _src, OutputArray _dst, InputArray _mask, OutputArray _validCount)
{
    Mat src = _src.getMat();
    
    // Make sure the input Mat object has a depth of CV_8U.
    CV_Assert( CV_8U == src.depth() );

    const int channels = src.channels();

    // Create a new Mat for the output.
    Mat dst;
    if ( 3 == channels )
    {
        // std::cout << "mNumKernals = " << mNumKernels << std::endl;
        _dst.create( mNumKernels, mNumKernels, CV_32FC3 );
        dst = _dst.getMat();
    }
    else if ( 1 == channels )
    {
        _dst.create( mNumKernels, mNumKernels, CV_32FC1 );
        dst = _dst.getMat();
    }
    else
    {
        // Error!
        EXCEPTION_BASE( "Mat with only 1 or 3 channels is supported." );
    }

    // Clear data in dst.
    dst.setTo( Scalar::all(0.0) );

    // Loop over every individual pixel in _src.
    uchar* pS = NULL;
    float* pD = NULL;
    int kernelIndexRow = 0, kernelIndexCol = 0;
    int pos = 0;
    int* const knlIdxRow = mKnlIdxRow.data();
    int* const knlIdxCol = mKnlIdxCol.data();

    // Mask.
    Mat mask = _mask.getMat();
    uchar* pM = NULL;
    _validCount.create( mNumKernels, mNumKernels, CV_8UC1 );
    Mat validCount = _validCount.getMat();
    validCount.setTo(Scalar::all(0));
    uchar* pV = NULL;

    for ( int i = 0; i < src.rows; ++i )
    {
        pS = src.ptr<uchar>(i);

        // Pointer to the dst Mat.
        kernelIndexRow = *( knlIdxRow + pos );
        pD = dst.ptr<R_t>( kernelIndexRow );

        // Pointer to the mask.
        pM = mask.ptr<uchar>(i);
        pV = validCount.ptr<uchar>(kernelIndexRow);

        for ( int j = 0; j < src.cols; ++j )
        {
            if ( 0 != *(pM + j) )
            {
                kernelIndexCol = *( knlIdxCol + pos );

                for ( int k = 0; k < channels; ++k )
                {
                    (*( pD + kernelIndexCol * channels + k )) += 
                    (*( pS +              j * channels + k ));
                }

                *(pV + kernelIndexCol) += 1;
            }

            pos++;
        }
    }

    validCount.copyTo(mTvcMat);

    for ( int i = 0; i < mNumKernels; ++i )
    {
        pV = mTvcMat.ptr<uchar>(i);

        for ( int j = 0; j < mNumKernels; ++j )
        {
            if ( 0 == *(pV + j) )
            {
                *(pV + j) = 1;
            }
        }
    }

    // Calculate the average.
    // dst /= tempValidCount;

    mat_divide<R_t, uchar, R_t>(dst, mTvcMat, dst);
}

void WeightColor::wc(const Mat& src, const Mat& mask, FMatrix_t& wc, Mat& avgColor)
{
    // Calculate the average color values.
    put_average_color_values( src, avgColor, mask, mVcMat );

    // NOTE: wc has to be row-major to maintain the performance.
    Real_t* pAvgColorVal = NULL;
    int pos              = 0;
    int posCol           = 0;
    const int channels   = src.channels();

    Real_t colorDiff     = 0.0;
    Real_t colorDist     = 0.0; // Color distance. L2 distance.

    Real_t colorSrc[3]   = {0.0, 0.0, 0.0};
    int centerIdx        = (mNumKernels - 1) / 2;

    Real_t* pWC          = wc.data();

    uchar* pVC           = NULL;

    pAvgColorVal = avgColor.ptr<Real_t>( centerIdx );
    for ( int i = 0; i < channels; ++i )
    {
        colorSrc[i] = *( pAvgColorVal + centerIdx*channels + i );
    }

    for ( int i = 0; i < avgColor.rows; ++i )
    {
        pAvgColorVal = avgColor.ptr<Real_t>( i );
        posCol       = 0;
        
        pVC = mVcMat.ptr<uchar>(i);

        for ( int j = 0; j < avgColor.cols; ++j )
        {
            if ( 0 != *(pVC + j) )
            {
                colorDist = 0.0;

                for ( int k = 0; k < channels; ++k )
                {
                    colorDiff = 
                        colorSrc[k] - *( pAvgColorVal + posCol + k );
                    
                    colorDist += colorDiff * colorDiff;
                }

                colorDist = std::sqrt( colorDist );

                *( pWC + pos ) = std::exp( -colorDist / mGammaC );
            }
            else
            {
                *( pWC + pos ) = 0.0;
            }

            posCol += channels;
            pos++;
        }
    }
}
