#include <boost/filesystem.hpp>
#include <cmath>
#include <fstream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "SLFusion/BilateralWindowMatcher.hpp"
#include "SLFusion/SLCommon.hpp"

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
    // mWss = (mPntDistKnl.array() / (-mGammaS)).exp().square().matrix();
    mWss = mWsMap.array().square().matrix();
}

BilateralWindowMatcher::BilateralWindowMatcher(int w, int nw)
: OCV_F_TYPE(CV_32FC1),
  mIM(w, nw),
  mGammaS(19), mGammaC(23), mTAD_T(125),
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
    // mWss         = FMatrix_t(mNumKernels, mNumKernels);
    mWss         = FMatrix_t(mWindowWidth, mWindowWidth);

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

    const int length = mWindowWidth;

    mACArrayRef = new Mat[size];
    create_mat_array( mACArrayRef, size, length, length, matType );
    mACArrayTst = new Mat[size];
    create_mat_array( mACArrayTst, size, length, length, matType );

    mWCArrayRef = new FM_t[size];
    create_matrix_array( mWCArrayRef, size, length, length );
    mWCArrayTst = new FM_t[size];
    create_matrix_array( mWCArrayTst, size, length, length );

    mPixelIdxRef = new int[size];
    mPixelIdxTst = new int[size];

    mABMemorySize = ( sizeOfMatEle + sizeof(R_t) + sizeof(int) ) * length * length * size * 2;
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

static void test_create_directory(const std::string& dn)
{
    namespace fs = boost::filesystem;

    fs::path d(dn);

    if ( false == fs::is_directory(d) )
    {
        // Create the directory.
        fs::create_directories(d);
        std::cout << "Directory " << dn << " is not exist. Create new directory." << std::endl;
    }
}

void BilateralWindowMatcher::debug_set_out_dir(const std::string& dir)
{
    mDebug_OutDir = dir;

    test_create_directory(mDebug_OutDir);
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

int BilateralWindowMatcher::debug_get_next_index_avg_color(void)
{
    return mDebug_ACIdx.at( mDebug_ACIdx.size() - 1 );
}

int BilateralWindowMatcher::debug_get_size_index_avg_color(void)
{
    return mDebug_ACIdx.size();
}

void BilateralWindowMatcher::debug_pop_index_avg_color(void)
{
    mDebug_ACIdx.pop_back();
}

void BilateralWindowMatcher::debug_push_index_avg_color(int idx)
{
    mDebug_ACIdx.push_back( idx );
}

template <typename _T, typename _D> 
static void write_mat_matlab_format(const std::string& path, const std::string& name, const Mat& m, int iFlag = 0, int w = 3)
{
    const int channels = m.channels();
    Mat* cArray = new Mat[channels];

    // Split m into separated channels.
    split( m, cArray );

    // describe_ocv_mat( cArray[0], "cArray[0]" );

    // Test the path.
    test_create_directory(path);

    // Prepare filename and output file stream.
    std::stringstream ssFn;
    std::ofstream ofs;
    const _T* p = NULL; // The pointer to the row header of splited channel.

    // Save the files.
    for ( int c = 0; c < channels; ++c )
    {
        // Filename.
        ssFn.str(""); ssFn.clear(); ssFn << path << "/" << name << "_" << c << ".dat";
        ofs.open( ssFn.str() );
        if ( 0 == iFlag )
        {
            ofs.precision(w);
        }

        // Loop over every element of one channel.
        for ( int i = 0; i < m.rows; ++i )
        {
            p = cArray[c].ptr<_T>(i);

            for ( int j = 0; j < m.cols; ++j )
            {
                if ( 0 == iFlag )
                {
                    ofs << std::scientific << (_D)( p[j] ) << " ";
                }
                else
                {
                    ofs << std::setw(w) << (_D)( p[j] ) << " ";
                }
                
            }

            ofs << std::endl;
        }

        ofs.close();
    }

    delete [] cArray; cArray = NULL;
}

#define DESCRIBE_OCV_MAT(M) \
    describe_ocv_mat( M, #M )

void BilateralWindowMatcher::debug_in_loop_wc_avg_color( int i,
        const Mat& ACArrayRef, const Mat& ACArrayTst,
        const Mat& refMat, const Mat& tstMat, const Mat& refMask, const Mat& tstMask,
        const FMatrix_t& WCRef, const FMatrix_t& WCTst)
{
    // Split ACArrayRef and ACArrayTst into separated channels.
    Mat channelsRef[3], channelsTst[3];
    split( ACArrayRef, channelsRef );
    split( ACArrayTst, channelsTst );

    // Prepare saving path.
    std::stringstream ssPath;
    ssPath << mDebug_OutDir << "/ac_" << std::setfill('0') << std::setw(4) << i;
    test_create_directory( ssPath.str() );
    std::string fnTemp  = ssPath.str() + "/mACArray.yml"; 

    // Save averaged color values, masks, and the original input windows into a yaml file.
    FileStorage fs;
    fs.open(fnTemp, FileStorage::WRITE);
    fs << "i" << i
        << "refC0" << channelsRef[0]
        << "refC1" << channelsRef[1]
        << "refC2" << channelsRef[2]
        << "tstC0" << channelsTst[0]
        << "tstC1" << channelsTst[1]
        << "tstC2" << channelsTst[2]
        << "refMask" << refMask
        << "tstMask" << tstMask
        << "refMat" << refMat
        << "tstMat" << tstMat;

    // Save matlab format files.
    // DESCRIBE_OCV_MAT(refMat);
    write_mat_matlab_format<uchar, int>( ssPath.str(), "windowRef", refMat, 1, 3 );
    write_mat_matlab_format<uchar, int>( ssPath.str(), "windowTst", tstMat, 1, 3 );
    write_mat_matlab_format<uchar, int>( ssPath.str(), "maskRef", refMask, 1, 3 );
    write_mat_matlab_format<uchar, int>( ssPath.str(), "maskTst", tstMask, 1, 3 );
    write_mat_matlab_format<R_t, R_t>( ssPath.str(), "acRef", ACArrayRef );
    write_mat_matlab_format<R_t, R_t>( ssPath.str(), "acTst", ACArrayTst );

    // Save the two input windows into image files.
    std::vector<int> jpegParams{ IMWRITE_JPEG_QUALITY, 100 };
    fnTemp = ssPath.str() + "/windowsRef.jpg";
    imwrite(fnTemp, refMat, jpegParams);
    fnTemp = ssPath.str() + "/windowsTst.jpg";
    imwrite(fnTemp, tstMat, jpegParams);
    
    // Save weight color values into plain text files.
    fnTemp = ssPath.str() + "/WCRef.dat";
    std::ofstream ofs;

    ofs.open( fnTemp );
    ofs << WCRef << std::endl;
    ofs.close();

    fnTemp = ssPath.str() + "/WCTst.dat";
    ofs.open( fnTemp );
    ofs << WCTst << std::endl;
    ofs.close();
}

void BilateralWindowMatcher::debug_in_loop_cost(int i, 
        Real_t cost, int disp, int idxAvgColorTst, 
        const Mat& ACRef, const Mat& ACTst,
        const FMatrix_t& WCRef, const FMatrix_t& WCTst,
        const FMatrix_t& tad)
{
    // Prepare the base path string and create the directory.
    std::stringstream ssPath;
    ssPath << mDebug_OutDir << "/lc_" 
           << std::setfill('0') << std::setw(4) << i << "/" 
           << std::setfill('0') << std::setw(4) << disp;
    test_create_directory( ssPath.str() );

    // Save the index, cost, and disparity values to a yaml file.
    std::string fnTemp;
    fnTemp = ssPath.str() + "/mACArray_cost.yml";
    FileStorage fs;
    fs.open(fnTemp, FileStorage::WRITE);
    fs << "i" << i
        << "tempCost" << cost
        << "disp" << disp
        << "idxAvgColorArrayTst" << idxAvgColorTst
        << "ref" << ACRef
        << "tst" << ACTst;

    // Save ACRef and ACTst into both MATLAB and JPEG formats.
    write_mat_matlab_format<R_t, R_t>( ssPath.str(), "ACRef", ACRef );
    write_mat_matlab_format<R_t, R_t>( ssPath.str(), "ACTst", ACTst );
    write_floating_point_mat_as_byte( ssPath.str() + "/ACRef", ACRef );
    write_floating_point_mat_as_byte( ssPath.str() + "/ACTst", ACTst );

    // Save tad.
    fnTemp = ssPath.str() + "/tad.dat";
    std::ofstream ofs;
    ofs.open( fnTemp );
    ofs << tad;
    ofs.close();

    // Save WCRef.
    fnTemp = ssPath.str() + "/WCRef.dat";
    ofs.open( fnTemp );
    ofs << WCRef << std::endl;
    ofs.close();

    // Save WCTst.
    fnTemp = ssPath.str() + "/WCTst.dat";
    ofs.open( fnTemp );
    ofs << WCTst << std::endl;
    ofs.close();

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
    Mat ac;
    FM_t wc(mNumKernels, mNumKernels);

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
        mWCO.wc( windowRef, winMaskRef, wc, ac );
        expand_block_2_window_mat<R_t>( ac, mACArrayRef[i] );
        expand_block_2_window_matrix<R_t>( wc, mWCArrayRef[i] );
        mWCO.wc( windowTst, winMaskTst, wc, ac );
        expand_block_2_window_mat<R_t>( ac, mACArrayTst[i] );
        expand_block_2_window_matrix<R_t>( wc, mWCArrayTst[i] );

        mPixelIdxRef[i] = idxRef;
        mPixelIdxTst[i] = idxTst;

        // Update indices.
        idxRef++;
        idxTst++;
    
        // Debug.
        if ( true == mFlagDebug && debug_get_size_index_avg_color() > 0 )
        {
            if ( i == debug_get_next_index_avg_color() )
            {
                std::cout << "Debug AC, i = " << i << std::endl;

                debug_in_loop_wc_avg_color( i,
                    mACArrayRef[i], mACArrayTst[i],
                    windowRef, windowTst, winMaskRef, winMaskTst,
                    mWCArrayRef[i], mWCArrayTst[i] );

                debug_pop_index_avg_color();
            }
        }
    }

    // FM_t tad( mNumKernels, mNumKernels );
    FM_t tad( mWindowWidth, mWindowWidth );
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
                if ( i == mDebug_ABIdx0 )
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