#include <cmath>
#include <fstream>
#include <vector>

#include <opencv2/highgui.hpp>

#include "SLFusion/BilateralWindowMatcher.hpp"

using namespace cv;
using namespace Eigen;
using namespace slf;

typedef BilateralWindowMatcher::IMatrix_t IM_t;
typedef BilateralWindowMatcher::FMatrix_t FM_t;
typedef BilateralWindowMatcher::Real_t    R_t;

static void
put_index_map(IM_t& ri, IM_t& ci, 
    IM_t& rri, IM_t& rci, 
    IM_t& knlRfnIdxRow, IM_t& knlRfnIdxCol, 
    int w)
{
    // Check compatibility between m and w.
    if ( ri.rows() != ri.cols() )
    {
        std::cout << "put_index_map: ri is not a square matrix." << std::endl;
        EXCEPTION_BAD_ARGUMENT(ri, "Not a square matrix.");
    }

    if ( ci.rows() != ci.cols() )
    {
        std::cout << "put_index_map: ci is not a square matrix." << std::endl;
        EXCEPTION_BAD_ARGUMENT(ci, "Not a square matrix.");
    }

    if ( rri.rows() != rri.cols() )
    {
        std::cout << "put_index_map: rri is not a square matrix." << std::endl;
        EXCEPTION_BAD_ARGUMENT(rri, "Not a square matrix.");
    }

    if ( rci.rows() != rci.cols() )
    {
        std::cout << "put_index_map: rci is not a square matrix." << std::endl;
        EXCEPTION_BAD_ARGUMENT(rci, "Not a square matrix.");
    }

    if ( knlRfnIdxRow.rows() != knlRfnIdxRow.cols() )
    {
        std::cout << "put_index_map: knlRfnIdxRow is not a square matrix." << std::endl;
        EXCEPTION_BAD_ARGUMENT(knlRfnIdxRow, "Not a square matrix.");
    }

    if ( knlRfnIdxCol.rows() != knlRfnIdxCol.cols() )
    {
        std::cout << "put_index_map: knlRfnIdxCol is not a square matrix." << std::endl;
        EXCEPTION_BAD_ARGUMENT(knlRfnIdxCol, "Not a square matrix.");
    }

    if ( ri.rows() != ci.rows() )
    {
        std::cout << "put_index_map: The dimensions of ri and ci are not the same." << std::endl;
        EXCEPTION_BAD_ARGUMENT(ri, "Dimensions of ri and ci are not compatible.");
    }

    if ( rri.rows() != rci.rows() )
    {
        std::cout << "put_index_map: The dimensions of rri and rci are not the same." << std::endl;
        EXCEPTION_BAD_ARGUMENT(rri, "Dimensions of rri and rci are not compatible.");
    }

    if ( knlRfnIdxRow.rows() != knlRfnIdxCol.rows() )
    {
        std::cout << "put_index_map: The dimensions of knlRfnIdxRow and knlRfnIdxCol are not the same." << std::endl;
        EXCEPTION_BAD_ARGUMENT(knlRfnIdxRow, "Dimensions of rri and rci are not compatible.");
    }

    if ( ri.rows() != rri.rows() )
    {
        std::cout << "put_index_map: The dimensions of ri and rri are not the same." << std::endl;
        EXCEPTION_BAD_ARGUMENT(ri, "Dimensions of ri and rri are not compatible.");
    }

    if ( 0x01 & w == 0x00 || w <= 0 )
    {
        std::cout << "put_index_map: w should be a positive odd integer." << std::endl;
        EXCEPTION_BAD_ARGUMENT(w, "Not a positive odd integer.");
    }

    if ( ri.rows() % w != 0 )
    {
        std::cout << "put_index_map: Dimensions of ri are not compatible with w." << std::endl;
        EXCEPTION_BAD_ARGUMENT(ri, "Dimensions not compatible with w.");
    }

    if ( ri.rows() / knlRfnIdxRow.rows() != w )
    {
        std::cout << "put_index_map: Dimensions of ri and  nklRfnIdxRow are not compatible." << std::endl;
        EXCEPTION_BAD_ARGUMENT(ri, "Dimensions of ri and  nklRfnIdxRow are not compatible.");
    }

    // Argument check done.

    // Kernel index.
    int si  = (w - 1) / 2; // Starting index.
    int k_ir = si, k_ic = si; // Kernel index row, kernel index col.

    int r_ir = 0, r_ic = 0; // Reference index row, reference index col.

    int rowCount = 0, colCount = 0;

    for ( int r = 0; r < ri.rows(); ++r )
    {
        if ( rowCount == 3 )
        {
            k_ir += w;
            r_ir++;

            rowCount = 0;
        }

        colCount = 0;
        k_ic     = si;
        r_ic     = 0;

        for ( int c = 0; c < ri.cols(); ++c )
        {
            if ( colCount == 3 )
            {
                k_ic += w;
                r_ic++;

                colCount = 0;
            }

            ri(r, c) = k_ir;
            ci(r, c) = k_ic;

            rri(r, c) = r_ir;
            rci(r, c) = r_ic;

            knlRfnIdxRow( r_ir, r_ic ) = k_ir;
            knlRfnIdxCol( r_ir, r_ic ) = k_ic;

            colCount++;
        }

        rowCount++;
    }
}

static void
put_distance_map(FM_t& rm, const IM_t& knlPntIdxRowMap, const IM_t& knlPntIdxColMap)
{
    // No dimension check here.

    // Get the original index of the center of the central kernal.
    int cntPos = ( knlPntIdxRowMap.rows() - 1 ) / 2;

    const int cntRow = knlPntIdxRowMap( cntPos, cntPos );
    const int cntCol = knlPntIdxColMap( cntPos, cntPos );

    rm = ( 
          (knlPntIdxRowMap.array() - cntRow).pow(2) 
        + (knlPntIdxColMap.array() - cntRow).pow(2) 
        ).sqrt().matrix().cast<R_t>();
}

static void
put_Ws_map( const FM_t& distanceMap, double gs, FM_t& Ws)
{
    Ws = (-1.0 * distanceMap / gs).array().exp().matrix();
}

BilateralWindowMatcher::BilateralWindowMatcher(int w, int nw)
: OCV_F_TYPE(CV_32FC1),
  mGammaS(1), mGammaC(5), mTAD_T(10000),
  mACArrayRef(NULL), mACArrayTst(NULL), mWCArrayRef(NULL), mWCArrayTst(NULL), mABSize(0),
  mABMemorySize(0),
  mFlagDebug(false), mDebug_ABIdx0(0), mDebug_ABIdx1(0), mDebug_OutDir("./DebugOutDir")
{
    // Check the validity of w and nw.
    if ( 0x01 & w == 0x00 || w <= 0)
    {
        std::cout << "w should be a positive odd integer. w = " << w << "." << std::endl;
        EXCEPTION_BAD_ARGUMENT(w, "A positive odd integer is expected.");
    }

    if ( 0x01 & nw == 0x00 || nw <= 0 )
    {
        std::cout << "nw should be a positive odd integer. nw = " << nw << "." << std::endl;
        EXCEPTION_BAD_ARGUMENT(nw, "A positive odd integer is expected.");
    }

    if ( w > 15 )
    {
        std::cout << "Currently the kernal size should be smaller than 15. w = " << w << "." << std::endl;
        EXCEPTION_BAD_ARGUMENT(w, "w should be smaller than 15.");

        // This is due to the mask mechanismt which is utilized in the process of calculating
        // the average color inside a window. The counter for unmasked pixels is a unsigned char typed value.
    }

    mKernelSize = w;
    mNumKernels = nw;

    // Create the index map and distance map.
    mWindowWidth = nw * w;

    mIndexMapRow  = IMatrix_t(mWindowWidth, mWindowWidth);
    mIndexMapCol  = IMatrix_t(mWindowWidth, mWindowWidth);
    mKnlIdxRow    = IMatrix_t(mWindowWidth, mWindowWidth);
    mKnlIdxCol    = IMatrix_t(mWindowWidth, mWindowWidth);
    mPntIdxKnlRow = IMatrix_t(mNumKernels, mNumKernels);
    mPntIdxKnlCol = IMatrix_t(mNumKernels, mNumKernels);

    mDistanceMap = FMatrix_t(mWindowWidth, mWindowWidth);
    mWsMap       = FMatrix_t(mWindowWidth, mWindowWidth);
    mWss         = FMatrix_t(mNumKernels, mNumKernels);

    mPntDistKnl  = FMatrix_t(mNumKernels, mNumKernels);

    // Put index maps.
    put_index_map( mIndexMapRow, mIndexMapCol, mKnlIdxRow, mKnlIdxCol, mPntIdxKnlRow, mPntIdxKnlCol, w );

    // Put distance map.
    put_distance_map( mDistanceMap, mIndexMapRow, mIndexMapCol );
    put_Ws_map( mDistanceMap, mGammaS, mWsMap );

    // Put point distance of kernels.
    int idxRow, idxCol;
    for ( int i = 0; i < nw; i++ )
    {
        for ( int j = 0; j < nw; j++ )
        {
            idxRow = mPntIdxKnlRow(i, j);
            idxCol = mPntIdxKnlCol(i, j);

            mPntDistKnl( i, j ) = mDistanceMap( idxRow, idxCol );
        }
    }

    // Calculate mWss.
    mWss = (mPntDistKnl.array() / (-mGammaS)).exp().square().matrix();
}

BilateralWindowMatcher::~BilateralWindowMatcher()
{
    destroy_array_buffer();
}

void BilateralWindowMatcher::show_index_maps(void)
{
    std::cout << "mKernelSize = " << mKernelSize << std::endl;
    std::cout << "mNumKernels = " << mNumKernels << std::endl;

    std::cout << "Row index map: " << std::endl;
    std::cout << mIndexMapRow << std::endl;

    std::cout << "Column index map: " << std::endl;
    std::cout << mIndexMapCol << std::endl;

    std::cout << "Kernel index row: " << std::endl;
    std::cout << mKnlIdxRow << std::endl;

    std::cout << "Kernel index column: " << std::endl;
    std::cout << mKnlIdxCol << std::endl;

    std::cout << "Point index of the kernel, row: " << std::endl;
    std::cout << mPntIdxKnlRow << std::endl;

    std::cout << "Point index of the kernel, column: " << std::endl;
    std::cout << mPntIdxKnlCol << std::endl;

    std::cout << "Distance map: " << std::endl;
    std::cout << mDistanceMap << std::endl;

    std::cout << "Point distance of kernel: " << std::endl;
    std::cout << mPntDistKnl << std::endl;

    std::cout << "Ws map: " << std::endl;
    std::cout << mWsMap << std::endl;

    std::cout << "Wss: " << std::endl;
    std::cout << mWss << std::endl;
}

int BilateralWindowMatcher::get_kernel_size(void)
{
    return mKernelSize;
}

int BilateralWindowMatcher::get_num_kernels_single_side(void)
{
    return mNumKernels;
}

int BilateralWindowMatcher::get_window_width(void)
{
    return mWindowWidth;
}

void BilateralWindowMatcher::set_gamma_s(Real_t gs)
{
    mGammaS = gs;
}

R_t 
BilateralWindowMatcher::get_gamma_s(void)
{
    return mGammaS;
}

void BilateralWindowMatcher::set_gamma_c(Real_t gc)
{
    mGammaC = gc;
}

R_t 
BilateralWindowMatcher::get_gamma_c(void)
{
    return mGammaC;
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

void BilateralWindowMatcher::put_average_color_values( 
    InputArray _src, OutputArray _dst, InputArray _mask, OutputArray _validCount, OutputArray _tempValidCount )
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

    _tempValidCount.create( mNumKernels, mNumKernels, CV_8UC1 );
    Mat tempValidCount = _tempValidCount.getMat();
    validCount.copyTo(tempValidCount);

    for ( int i = 0; i < mNumKernels; ++i )
    {
        pV = tempValidCount.ptr<uchar>(i);

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

    mat_divide<R_t, uchar, R_t>(dst, tempValidCount, dst);
}

void BilateralWindowMatcher::put_wc(const Mat& src, const Mat& mask, 
    FMatrix_t& wc, Mat& avgColor, Mat& vcMat, Mat& tvcMat, Mat* bufferS)
{
    // To see if we have external buffer provided.
    bool tempBufferS = false;

    if ( NULL == bufferS )
    {
        // Overwrite the input argument!
        bufferS = new Mat( mWindowWidth, mWindowWidth, OCV_F_TYPE );
        tempBufferS = true;
    }

    // Calculate the average color values.
    put_average_color_values( src, avgColor, mask, vcMat, tvcMat );

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
        
        pVC = vcMat.ptr<uchar>(i);

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

    // Release the memory.
    if ( true == tempBufferS )
    {
        delete bufferS; bufferS = NULL;
    }
}

template<typename tR, typename tT> 
BilateralWindowMatcher::Real_t BilateralWindowMatcher::TAD( const tR* pr, const tT* pt, int channels )
{
    Real_t temp = 0.0;

    for ( int i = 0; i < channels; ++i )
    {
        temp += ( pr[i] - pt[i] ) * ( pr[i] - pt[i] );
    }

    return (Real_t)( 
        std::min( std::fabs( temp ), mTAD_T ) 
        );
}

template<typename _TR, typename _TT> 
void BilateralWindowMatcher::TADm(const Mat& ref, const Mat& tst, FMatrix_t& tad)
{
    // The rows and cols of ref and tst are assumed to be the same.
    const int channels = ref.channels();
    const _TR* pRef  = NULL;
    const _TT* pTst  = NULL;

    R_t temp = 0.0;

    int posCol = 0, posColShift = 0;
    int posTad = 0;

    tad.setConstant(0.0);

    for ( int i = 0; i < ref.rows; ++i )
    {
        // Get the pointer to the ref and tst.
        pRef = ref.ptr<_TR>(i);
        pTst = tst.ptr<_TT>(i);

        posCol = 0;

        for ( int j = 0; j < ref.cols; ++j )
        {
            if ( i == ref.rows/2 && j == ref.cols/2 )
            {
                temp = 0.0;
            }

            temp        = 0.0;
            posColShift = posCol;

            for ( int k = 0; k < channels; ++k)
            {
                temp += 
                ( pRef[posColShift] - pTst[posColShift] ) * 
                ( pRef[posColShift] - pTst[posColShift] );

                posColShift++;
            }

            *(tad.data() + posTad) = std::min( std::sqrt(temp), mTAD_T );

            posTad++;
            posCol += channels;
        }
    }
}

void BilateralWindowMatcher::destroy_array_buffer(void)
{
    delete [] mWCArrayTst; mWCArrayTst = NULL; // Delete a NULL pointer won't cause problem.
    delete [] mWCArrayRef; mWCArrayRef = NULL;
    delete [] mACArrayTst; mACArrayTst = NULL;
    delete [] mACArrayRef; mACArrayRef = NULL;
}

static void 
create_mat_array(Mat* array, size_t size, int row, int col, int type)
{
    for ( size_t i = 0; i < size; ++i )
    {
        array[i].create( row, col, type );
    }
}

template<typename _T>
static void 
create_matrix_array(_T* array, size_t size, int row, int col)
{
    for ( size_t i = 0; i < size; ++i )
    {
        array[i] = _T::Zero( row, col );
    }
}

void BilateralWindowMatcher::allocate_array_buffer(size_t size, int matType)
{
    size_t sizeOfMatEle = 0;
    switch (matType)
    {
        case CV_32FC1:
        {
            sizeOfMatEle = sizeof(float);
            break;
        }
        case CV_32FC3:
        {
            sizeOfMatEle = sizeof(float) * 3;
            break;
        }
        case CV_64FC1:
        {
            sizeOfMatEle = sizeof(double);
            break;
        }
        case CV_64FC3:
        {
            sizeOfMatEle = sizeof(double) * 3;
            break;
        }
        case CV_8UC1:
        {
            sizeOfMatEle = sizeof(uchar);
            break;
        }
        case CV_8UC3:
        {
            sizeOfMatEle = sizeof(uchar) * 3;
            break;
        }
        default:
        {
            // This should be an error.
            EXCEPTION_BAD_ARGUMENT(matType, "Unexpected matType,");
        }
    }

    mACArrayRef = new Mat[size];
    create_mat_array( mACArrayRef, size, mNumKernels, mNumKernels, matType );
    mACArrayTst = new Mat[size];
    create_mat_array( mACArrayTst, size, mNumKernels, mNumKernels, matType );

    mWCArrayRef = new FM_t[size];
    create_matrix_array( mWCArrayRef, size, mNumKernels, mNumKernels );
    mWCArrayTst = new FM_t[size];
    create_matrix_array( mWCArrayTst, size, mNumKernels, mNumKernels );

    mABMemorySize = ( sizeOfMatEle + sizeof(R_t) ) * mNumKernels * mNumKernels * size;
}

size_t BilateralWindowMatcher::get_internal_buffer_szie(void)
{
    return mABMemorySize;
}

void BilateralWindowMatcher::enable_debug(void)
{
    mFlagDebug = true;
}

void BilateralWindowMatcher::disable_debug(void)
{
    mFlagDebug = false;
}

void BilateralWindowMatcher::debug_set_array_buffer_idx(size_t idx0, size_t idx1)
{
    mDebug_ABIdx0 = idx0;
    mDebug_ABIdx1 = idx1;
}

void BilateralWindowMatcher::debug_set_out_dir(const std::string& dir)
{
    mDebug_OutDir = dir;
}

void BilateralWindowMatcher::create_array_buffer(size_t size, int matType, bool force)
{
    if ( size > mABSize )
    {
        destroy_array_buffer();
        allocate_array_buffer(size, matType);
    }
    else if ( size == 0 )
    {
        destroy_array_buffer();
    }
    else if ( size <= mABSize )
    {
        if ( true == force )
        {
            destroy_array_buffer();
            allocate_array_buffer(size, matType);
        }
    }

    mABSize = size;
}

void BilateralWindowMatcher::match_single_line(
        const Mat& refMat, const Mat& tstMat, 
        const Mat& refMask, const Mat& tstMask, 
        int rowIdx,
        int minDisp, int maxDisp, 
        MatchingCost<BilateralWindowMatcher::Real_t>* pMC, int* nMC )
{
    if ( true == mFlagDebug )
    {
        std::vector<int> jpegParams;
        jpegParams.push_back(IMWRITE_JPEG_QUALITY);
        jpegParams.push_back(100);

        imwrite("refMat.jpg",   refMat, jpegParams);
        imwrite("tstMat.jpg",   tstMat, jpegParams);
        imwrite("refMask.jpg", refMask, jpegParams);
        imwrite("tstMask.jpg", tstMask, jpegParams);
    }

    // Expecting input images have the same width.
    if ( refMat.cols != tstMat.cols )
    {
        std::stringstream ssRef;
        ssRef << "( " << refMat.cols << ", " << refMat.rows << " )";

        std::stringstream ssTst;
        ssTst << "( " << tstMat.cols << ", " << tstMat.rows << " )";

        EXCEPTION_DIMENSION_MISMATCH(refMat, ssRef.str(), tstMat, ssTst.str());
    }

    if ( refMask.cols != tstMask.cols )
    {
        std::stringstream ssRef;
        ssRef << "( " << refMask.cols << ", " << refMask.rows << " )";

        std::stringstream ssTst;
        ssTst << "( " << tstMask.cols << ", " << tstMask.rows << " )";

        EXCEPTION_DIMENSION_MISMATCH(refMask, ssRef.str(), tstMask, ssTst.str());
    }

    if ( refMat.cols != refMask.cols )
    {
        std::stringstream ssRef;
        ssRef << "( " << refMat.cols << ", " << refMat.rows << " )";

        std::stringstream ssMask;
        ssMask << "( " << refMask.cols << ", " << refMask.rows << " )";

        EXCEPTION_DIMENSION_MISMATCH(refMat, ssRef.str(), refMask, ssMask.str());
    }

    if ( refMat.cols <= mWindowWidth + minDisp )
    {
        EXCEPTION_BAD_ARGUMENT( refMat, "Not enough columns." );
    }

    int halfCount = (mWindowWidth - 1) / 2;

    // The input images must have enough rows.
    if ( refMat.rows < mWindowWidth ||
         tstMat.rows < mWindowWidth ||
         rowIdx < halfCount ||
         rowIdx > ( refMat.rows - halfCount - 1 ) || 
         rowIdx > ( tstMat.rows - halfCount - 1 ) )
    {
        std::stringstream ss;
        ss << "refMat.rows (" << refMat.rows << "), " 
           << "tstMat.rows (" << tstMat.rows << "), " 
           << "and rowIdx (" << rowIdx << ") are not compatible.";

        EXCEPTION_BASE( ss.str() );
    }

    // minDisp and maxDisp.
    if ( minDisp <= 0 )
    {
        EXCEPTION_BAD_ARGUMENT(minDisp, "minDisp must be a positive number.");
    }

    if ( maxDisp >= refMat.cols )
    {
        EXCEPTION_BAD_ARGUMENT(maxDisp, "maxDisp must be smaller than the width of image.");
    }

    if ( minDisp >= maxDisp )
    {
        EXCEPTION_BASE("minDisp should be smaller than maxDisp.");
    }

    // pMC cannot be NULL.
    if ( NULL == pMC )
    {
        EXCEPTION_BAD_ARGUMENT( pMC, "NULL pointer found." );
    }

    // === Initial check done. ===

    const int numDisp = maxDisp - minDisp + 1;
    const int pixels  = refMat.cols - minDisp - halfCount * 2;

    const int nC      = pixels * numDisp;

    // === Pre-allocation of weights. ===
    // == Pre-allocation of average color. ===
    create_array_buffer(pixels, refMat.type());

    // === Calculate color weights for all valid pixels. ===

    int idxRef = minDisp + halfCount, idxTst = halfCount;
    Mat windowRef, windowTst;
    Mat winMaskRef, winMaskTst;
    Range rowRange( rowIdx - halfCount, rowIdx + halfCount + 1 );

    // Buffers.
    Mat bufferS( mWindowWidth, mWindowWidth, OCV_F_TYPE );
    Mat vcMatRef(  mNumKernels, mNumKernels, CV_8UC1 );
    Mat vcMatTst(  mNumKernels, mNumKernels, CV_8UC1 );
    Mat tvcMatRef( mNumKernels, mNumKernels, CV_8UC1 );
    Mat tvcMatTst( mNumKernels, mNumKernels, CV_8UC1 );

    for ( int i = 0; i < pixels; ++i )
    {
        // Update the ROIs in refMat and tstMat.
        Range colRangeRef( idxRef - halfCount, idxRef + halfCount + 1 );
        Range colRangeTst( idxTst - halfCount, idxTst + halfCount + 1 );

        windowRef = refMat( rowRange, colRangeRef );
        windowTst = tstMat( rowRange, colRangeTst );

        winMaskRef = refMask( rowRange, colRangeRef );
        winMaskTst = tstMask( rowRange, colRangeTst );

        // Calculate weight matrix.
        // avgColorArrayRef[i].create( mNumKernels, mNumKernels,  );
        // avgColorArrayTst[i].create( mNumKernels, mNumKernels,  );
        // Memory allocation for avgColorArrayRef and avgColorArrayTst will occur inside put_wc().
        put_wc( windowRef, winMaskRef, mWCArrayRef[i], mACArrayRef[i], vcMatRef, tvcMatRef, &bufferS );
        put_wc( windowTst, winMaskTst, mWCArrayTst[i], mACArrayTst[i], vcMatTst, tvcMatTst, &bufferS );

        // Update indices.
        idxRef++;
        idxTst++;
    
        // Debug.
        if ( true == mFlagDebug )
        {
            if ( i == mDebug_ABIdx0 )
            {
                Mat channelsRef[3], channelsTst[3];
                split( mACArrayRef[i], channelsRef );
                split( mACArrayTst[i], channelsTst );

                std::string fnTemp = mDebug_OutDir + "/mACArray.yml"; 

                FileStorage fs;
                fs.open(fnTemp, FileStorage::WRITE);
                fs << "i" << i
                   << "refC0" << channelsRef[0]
                   << "refC1" << channelsRef[1]
                   << "refC2" << channelsRef[2]
                   << "tstC0" << channelsTst[0]
                   << "tstC1" << channelsTst[1]
                   << "tstC2" << channelsTst[2]
                   << "winMaskRef" << winMaskRef
                   << "winMaskTst" << winMaskTst
                   << "vcMatRef" << vcMatRef
                   << "vcMatTst" << vcMatTst
                   << "tvbMatRef" << tvcMatRef
                   << "tvbMatTst" << tvcMatTst
                   << "windowRef" << windowRef
                   << "windowTst" << windowTst;

                std::vector<int> jpegParams;
                jpegParams.push_back(IMWRITE_JPEG_QUALITY);
                jpegParams.push_back(100);

                fnTemp = mDebug_OutDir + "/windowsRef.jpg";
                imwrite(fnTemp, windowRef, jpegParams);
                fnTemp = mDebug_OutDir + "/windowsTst.jpg";
                imwrite(fnTemp, windowTst, jpegParams);

                fnTemp = mDebug_OutDir + "/mWCArray.dat";
                std::ofstream ofs;
                ofs.open( fnTemp );
                ofs << mWCArrayRef[i] << std::endl << std::endl << mWCArrayTst[i];
                ofs.close();
            }
        }
    }

    FM_t tad( mNumKernels, mNumKernels );
    int idxAvgColorArrayTst = 0; // The index for avgColorArrayTst.
    FM_t tempDenominatorMatrix;
    R_t  tempCost = 0.0;

    // === Calculate the cost. ===
    int debugCount = 0;
    for ( int i = 0; i < pixels; ++i )
    {
        // The index in the original image.
        idxRef = halfCount + minDisp + i;

        pMC[i].set_idx_ref( idxRef );

        for ( int j = 0; j < numDisp; ++j )
        {
            // Do not allocate new memory for MachingCost objects.
            // These objects should be already allocated outside to boost
            // the overall performance.

            // The index in avgColorArrayTst.
            idxAvgColorArrayTst = i - j;

            if ( idxAvgColorArrayTst < 0 )
            {
                break;
            }

            if ( true == mFlagDebug )
            {
                if ( i == mDebug_ABIdx0 && j == mDebug_ABIdx1 )
                {
                    std::cout << "Debug." << std::endl;
                }
            }

            // Calculate the TAD over all the kernel blocks of windowRef and windowTst.
            TADm<R_t, R_t>( mACArrayRef[i], mACArrayTst[idxAvgColorArrayTst], tad );

            // Calculate the cost value.
            tempDenominatorMatrix = ( mWss.array() * mWCArrayRef[i].array() * mWCArrayTst[idxAvgColorArrayTst].array() ).matrix();

            tempCost = 
                ( tempDenominatorMatrix.array() * tad.array() ).sum() / 
                tempDenominatorMatrix.sum();

            // Save the cost value into pMC.
            pMC[i].push_back( minDisp + j, tempCost );

            if ( true == mFlagDebug )
            {
                if ( i == mDebug_ABIdx0 && j == mDebug_ABIdx1 )
                {
                    std::string fnTemp;

                    fnTemp = mDebug_OutDir + "/mACArray_cost.yml";
                    FileStorage fs;
                    fs.open(fnTemp, FileStorage::WRITE);
                    fs << "i" << i
                       << "tempCost" << tempCost
                       << "disp" << minDisp + j
                       << "idxAvgColorArrayTst" << idxAvgColorArrayTst
                       << "ref" << mACArrayRef[i]
                       << "tst" << mACArrayTst[idxAvgColorArrayTst];

                    fnTemp = mDebug_OutDir + "/mWss.dat";
                    std::ofstream ofs;
                    ofs.open( fnTemp );
                    ofs << mWss;
                    ofs.close();

                    fnTemp = mDebug_OutDir + "/tad.dat";
                    ofs.open( fnTemp );
                    ofs << tad;
                    ofs.close();

                    fnTemp = mDebug_OutDir + "/WCArrayCost.dat";
                    ofs.open( fnTemp );
                    ofs << "ref" << std::endl
                        << mWCArrayRef[i].array() << std::endl
                        << "tst" << std::endl
                        << mWCArrayTst[idxAvgColorArrayTst].array();

                    std::cout << "tempCost = " << tempCost << std::endl;
                }        
            }

            debugCount++;
        }

        // // Debug.
        // std::cout << "i = " << i << std::endl;
    }

    // Debug.
    std::cout << "Costs calculated." << std::endl;
}
