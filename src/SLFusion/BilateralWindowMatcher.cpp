
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
: mGammaS(14), mGammaC(23)
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

    mKernelSize = w;
    mNumKernels = nw;

    // Create the index map and distance map.
    int windowWidth = nw * w;

    mIndexMapRow  = IMatrix_t(windowWidth, windowWidth);
    mIndexMapCol  = IMatrix_t(windowWidth, windowWidth);
    mKnlIdxRow    = IMatrix_t(windowWidth, windowWidth);
    mKnlIdxCol    = IMatrix_t(windowWidth, windowWidth);
    mPntIdxKnlRow = IMatrix_t(nw, nw);
    mPntIdxKnlCol = IMatrix_t(nw, nw);

    mDistanceMap = FMatrix_t(windowWidth, windowWidth);
    mWsMap       = FMatrix_t(windowWidth, windowWidth);

    mPntDistKnl  = FMatrix_t(nw, nw);

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
}

BilateralWindowMatcher::~BilateralWindowMatcher()
{

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
}

int BilateralWindowMatcher::get_kernel_size(void)
{
    return mKernelSize;
}

int BilateralWindowMatcher::get_num_kernels_single_side(void)
{
    return mNumKernels;
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

void BilateralWindowMatcher::put_average_color_values(InputArray _src, OutputArray _dst)
{
    Mat src = _src.getMat();
    
    // Make sure the input Mat object has a depth of CV_8U.
    CV_Assert( CV_8U == src.depth() );

    const int channels = src.channels();

    // Create a new Mat for the output.
    Mat dst;
    if ( 3 == channels )
    {
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

    for ( int i = 0; i < src.rows; ++i )
    {
        pS = src.ptr<uchar>(i);

        // Pointer to the dst Mat.
        kernelIndexRow = *( knlIdxRow + pos );
        pD = dst.ptr<float>( kernelIndexRow );

        for ( int j = 0; j < src.cols; ++j )
        {
            kernelIndexCol = *( knlIdxCol + pos );

            for ( int k = 0; k < channels; ++k )
            {
                (*( pD + kernelIndexCol * channels + k )) += 
                (*( pS +              j * channels + k ));
            }

            pos++;
        }
    }

    // Calculate the average.
    dst /= mKernelSize * mKernelSize;
}

void BilateralWindowMatcher::put_wc(const Mat& src, FMatrix_t& wc)
{
    // Assuming that we are only work with Mat whoes depth is CV_8U.
}
