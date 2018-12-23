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

static void
put_distance_map(FM_t& rm, const IM_t& knlPntIdxRowMap, const IM_t& knlPntIdxColMap)
{
    // No dimension check here.

    // Get the original index of the center of the central kernal.
    int cntPos = ( knlPntIdxRowMap.rows() - 1 ) / 2;

    const int cntRow = knlPntIdxRowMap( cntPos, cntPos );
    const int cntCol = knlPntIdxColMap( cntPos, cntPos );

    rm = ( 
          (knlPntIdxRowMap.cast<R_t>().array() - cntRow).pow(2) 
        + (knlPntIdxColMap.cast<R_t>().array() - cntCol).pow(2) 
        ).sqrt().matrix();
}

static void
put_Ws_map( const FM_t& distanceMap, R_t gs, FM_t& Ws)
{
    Ws = (-1.0 * distanceMap / gs).array().exp().matrix();
}

void BilateralWindowMatcher::update_ws(void)
{
    put_Ws_map( mDistanceMap, mGammaS, mWsMap );

    // Put point distance of kernels.
    int idxRow, idxCol;
    for ( int i = 0; i < mNumKernels; i++ )
    {
        for ( int j = 0; j < mNumKernels; j++ )
        {
            idxRow = mIM.mPntIdxKnlRow(i, j);
            idxCol = mIM.mPntIdxKnlCol(i, j);

            mPntDistKnl( i, j ) = mDistanceMap( idxRow, idxCol );
        }
    }

    // Calculate mWss.
    mWss = (mPntDistKnl.array() / (-mGammaS)).exp().square().matrix();
}

BilateralWindowMatcher::BilateralWindowMatcher(int w, int nw)
: OCV_F_TYPE(CV_32FC1),
  mIM(w, nw),
  mGammaS(1), mGammaC(5), mTAD_T(10000),
  mACArrayRef(NULL), mACArrayTst(NULL), mWCArrayRef(NULL), mWCArrayTst(NULL), 
  mPixelIdxRef(NULL), mPixelIdxTst(NULL), mABSize(0),
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

    mKernelSize  = w;
    mNumKernels  = nw;
    mWindowWidth = mKernelSize * mNumKernels;

    // Create the index map and distance map.

    mDistanceMap = FMatrix_t(mWindowWidth, mWindowWidth);
    mWsMap       = FMatrix_t(mWindowWidth, mWindowWidth);
    mPntDistKnl  = FMatrix_t(mNumKernels, mNumKernels);
    mWss         = FMatrix_t(mNumKernels, mNumKernels);

    // Put distance map.
    put_distance_map( mDistanceMap, mIM.mIndexMapRow, mIM.mIndexMapCol );
    update_ws();

    // WeightColor object.
    mWCO = WeightColor( mNumKernels, mIM.mKnlIdxRow, mIM.mKnlIdxCol, mGammaC );
}

BilateralWindowMatcher::~BilateralWindowMatcher()
{
    destroy_array_buffer();
}

void BilateralWindowMatcher::show_index_maps(void)
{
    std::cout << "mKernelSize = " << mKernelSize << std::endl;
    std::cout << "mNumKernels = " << mNumKernels << std::endl;

    mIM.show();

    std::cout << "Distance map: " << std::endl;
    std::cout << mDistanceMap << std::endl;

    std::cout << "Point distance of kernel: " << std::endl;
    std::cout << mPntDistKnl << std::endl;

    std::cout << "Ws map: " << std::endl;
    std::cout << mWsMap << std::endl;

    std::cout << "Wss: " << std::endl;
    std::cout << mWss << std::endl;
}

int BilateralWindowMatcher::get_kernel_size(void) const
{
    return mKernelSize;
}

int BilateralWindowMatcher::get_num_kernels_single_side(void) const
{
    return mNumKernels;
}

int BilateralWindowMatcher::get_window_width(void) const
{
    return mWindowWidth;
}

void BilateralWindowMatcher::set_gamma_s(Real_t gs)
{
    mGammaS = gs;
    update_ws();
}

R_t 
BilateralWindowMatcher::get_gamma_s(void) const
{
    return mGammaS;
}

void BilateralWindowMatcher::set_gamma_c(Real_t gc)
{
    mGammaC = gc;
}

R_t BilateralWindowMatcher::get_gamma_c(void) const
{
    return mGammaC;
}

template<typename tR, typename tT> 
Real_t BilateralWindowMatcher::TAD( const tR* pr, const tT* pt, int channels )
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
    const _TR* pRef    = NULL;
    const _TT* pTst    = NULL;

    R_t temp = 0.0;

    int posCol = 0, posColShift = 0;
    int posTad = 0;
    // Clear tad.
    tad.setConstant(0.0);

    for ( int i = 0; i < ref.rows; ++i )
    {
        // Get the pointer to the ref and tst.
        pRef = ref.ptr<_TR>(i);
        pTst = tst.ptr<_TT>(i);

        posCol = 0;

        for ( int j = 0; j < ref.cols; ++j )
        {
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
    delete [] mPixelIdxTst; mPixelIdxTst = NULL;
    delete [] mPixelIdxRef; mPixelIdxRef = NULL;
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

    mPixelIdxRef = new int[size];
    mPixelIdxTst = new int[size];

    mABMemorySize = ( sizeOfMatEle + sizeof(R_t) + sizeof(int) ) * mNumKernels * mNumKernels * size * 2;
}

size_t BilateralWindowMatcher::get_internal_buffer_szie(void) const
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

static int half_count(int n)
{
    // Works only with odd number.
    if ( n & 0x01 != 0x01 )
    {
        std::stringstream ss;
        ss << "n must be an odd number. n = " << n;

        EXCEPTION_BAD_ARGUMENT(, ss.str());
    }

    // Works only with n > 2.
    if ( n <= 2 )
    {
        std::stringstream ss;
        ss << "n must be greater than 2. n = " << n;

        EXCEPTION_BAD_ARGUMENT(n, ss.str() );
    }

    return ( n - 1 ) / 2;
}

static int num_inner_pixels(int cols, int minDisp, int halfCount)
{
    return cols - minDisp - halfCount * 2;
}

void BilateralWindowMatcher::debug_in_loop_wc_avg_color( int i,
        const Mat& ACArrayRef, const Mat& ACArrayTst,
        const Mat& refMat, const Mat& tstMat, const Mat& refMask, const Mat& tstMask,
        const FMatrix_t& WCRef, const FMatrix_t& WCTst)
{
    Mat channelsRef[3], channelsTst[3];
    split( ACArrayRef, channelsRef );
    split( ACArrayTst, channelsTst );

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
        << "winMaskRef" << refMask
        << "winMaskTst" << tstMask
        << "windowRef" << refMat
        << "windowTst" << tstMat;

    std::vector<int> jpegParams;
    jpegParams.push_back(IMWRITE_JPEG_QUALITY);
    jpegParams.push_back(100);

    fnTemp = mDebug_OutDir + "/windowsRef.jpg";
    imwrite(fnTemp, refMat, jpegParams);
    fnTemp = mDebug_OutDir + "/windowsTst.jpg";
    imwrite(fnTemp, tstMat, jpegParams);

    fnTemp = mDebug_OutDir + "/mWCArray.dat";
    std::ofstream ofs;
    ofs.open( fnTemp );
    ofs << WCRef << std::endl << std::endl << WCTst;
    ofs.close();
}

void BilateralWindowMatcher::debug_in_loop_cost(int i, 
        Real_t cost, int disp, int idxAvgColorTst, 
        const Mat& ACRef, const Mat& ACTst,
        const FMatrix_t& WCRef, const FMatrix_t& WCTst,
        const FMatrix_t& tad)
{
    std::string fnTemp;

    fnTemp = mDebug_OutDir + "/mACArray_cost.yml";
    FileStorage fs;
    fs.open(fnTemp, FileStorage::WRITE);
    fs << "i" << i
        << "tempCost" << cost
        << "disp" << disp
        << "idxAvgColorArrayTst" << idxAvgColorTst
        << "ref" << ACRef
        << "tst" << ACTst;

    fnTemp = mDebug_OutDir + "/tad.dat";
    std::ofstream ofs;
    ofs.open( fnTemp );
    ofs << tad;
    ofs.close();

    fnTemp = mDebug_OutDir + "/WCArrayCost.dat";
    ofs.open( fnTemp );
    ofs << "ref" << std::endl
        << WCRef << std::endl
        << "tst" << std::endl
        << WCTst;

    std::cout << "tempCost = " << cost << std::endl;
}

void BilateralWindowMatcher::match_single_line(
        const Mat& refMat, const Mat& tstMat, 
        const Mat& refMask, const Mat& tstMask, 
        int rowIdx,
        int minDisp, int maxDisp, 
        MatchingCost<Real_t>* pMC, int* nMC )
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

    const int halfCount = half_count(mWindowWidth);

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
    // ===========================
    // === Initial check done. ===
    // ===========================

    const int numDisp = maxDisp - minDisp + 1;
    const int pixels  = num_inner_pixels(refMat.cols, minDisp, halfCount);
    const int nC      = pixels * numDisp; // Number of cost objects.

    // === Pre-allocation of weights. ===
    // == Pre-allocation of average color. ===
    create_array_buffer( pixels, CVType::get_real_number_type( refMat.type() ) );

    // =====================================================
    // === Calculate color weights for all valid pixels. ===
    // =====================================================

    int idxRef = minDisp + halfCount, idxTst = halfCount;
    Range rowRange( rowIdx - halfCount, rowIdx + halfCount + 1 );
    Range colRangeRef( idxRef - halfCount, idxRef + halfCount + 1 );
    Range colRangeTst( idxTst - halfCount, idxTst + halfCount + 1 );

    Mat windowRef, windowTst;
    Mat winMaskRef, winMaskTst;

    for ( int i = 0; i < pixels; ++i )
    {
        // Update the ROIs in refMat and tstMat.
        colRangeRef.start = idxRef - halfCount;
        colRangeRef.end   = idxRef + halfCount + 1;
        colRangeTst.start = idxTst - halfCount;
        colRangeTst.end   = idxTst + halfCount + 1;

        // Get the image patches and the masks.
        windowRef  = refMat( rowRange, colRangeRef );
        windowTst  = tstMat( rowRange, colRangeTst );
        winMaskRef = refMask( rowRange, colRangeRef );
        winMaskTst = tstMask( rowRange, colRangeTst );

        // Calculate weight matrix.

        // Memory allocation for avgColorArrayRef and avgColorArrayTst will occur inside mWCO.wc().
        mWCO.wc( windowRef, winMaskRef, mWCArrayRef[i], mACArrayRef[i] );
        mWCO.wc( windowTst, winMaskTst, mWCArrayTst[i], mACArrayTst[i] );

        mPixelIdxRef[i] = idxRef;
        mPixelIdxTst[i] = idxTst;

        // Update indices.
        idxRef++;
        idxTst++;
    
        // Debug.
        if ( true == mFlagDebug )
        {
            if ( i == mDebug_ABIdx0 )
            {
                debug_in_loop_wc_avg_color( i,
                    mACArrayRef[i], mACArrayTst[i],
                    windowRef, windowTst, winMaskRef, winMaskTst,
                    mWCArrayRef[i], mWCArrayTst[i] );
            }
        }
    }

    FM_t tad( mNumKernels, mNumKernels );
    int  idxAvgColorArrayTst = 0; // The index for avgColorArrayTst.
    FM_t tempDenominatorMatrix;
    R_t  tempCost = 0.0;

    // === Calculate the cost. ===
    int debugCount = 0;
    for ( int i = 0; i < pixels; ++i )
    {
        // The index in the original image.
        idxRef = mPixelIdxRef[i];
        // Save the current reference index.
        pMC[i].set_idx_ref( idxRef );
        pMC[i].reset();

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
                    std::cout << "Debug." << std::endl; // This line is for placing a breakpoint.
                }
            }

            // Calculate the TAD over all the kernel blocks of windowRef and windowTst.
            TADm<R_t, R_t>( mACArrayRef[i], mACArrayTst[idxAvgColorArrayTst], tad );

            // Calculate the cost value.
            tempDenominatorMatrix = ( mWss.array() * mWCArrayRef[i].array() * mWCArrayTst[idxAvgColorArrayTst].array() ).matrix();

            tempCost = 
                ( tempDenominatorMatrix.array() * tad.array() ).sum() / tempDenominatorMatrix.sum();

            // Save the cost value into pMC.
            pMC[i].push_back( idxRef - mPixelIdxTst[idxAvgColorArrayTst], tempCost );

            if ( true == mFlagDebug )
            {
                if ( i == mDebug_ABIdx0 && j == mDebug_ABIdx1 )
                {
                    debug_in_loop_cost(i, 
                        tempCost, idxRef - mPixelIdxTst[idxAvgColorArrayTst], idxAvgColorArrayTst,
                        mACArrayRef[i], mACArrayTst[idxAvgColorArrayTst],
                        mWCArrayRef[i], mWCArrayTst[idxAvgColorArrayTst],
                        tad);
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

 // ===================== Test section. ===========================

#ifdef TEST_SECTION

TEST_F( Test_BilateralWindowMatcher, half_count )
{
    ASSERT_EQ( half_count(39), 19 ) << "The half count of 39 should be 19.";
}

TEST_F( Test_BilateralWindowMatcher, inner_pixels )
{
    ASSERT_EQ( num_inner_pixels( 138, 20, 19 ), 80 ) << "The number of inner pixels.";
}

#endif