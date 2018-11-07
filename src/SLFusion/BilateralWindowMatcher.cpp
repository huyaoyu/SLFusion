
#include "SLFusion/BilateralWindowMatcher.hpp"

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using namespace slf;

static void
put_index_map(MatrixXi& ri, MatrixXi& ci, 
    MatrixXi& rri, MatrixXi& rci, 
    MatrixXi& knlRfnIdxRow, MatrixXi& knlRfnIdxCol, 
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
put_distance_map(MatrixXd& rm, const MatrixXi& knlPntIdxRowMap, const MatrixXi& knlPntIdxColMap)
{
    // No dimension check here.

    // Get the original index of the center of the central kernal.
    int cntPos = ( knlPntIdxRowMap.rows() - 1 ) / 2;

    int cntRow = knlPntIdxRowMap( cntPos, cntPos );
    int cntCol = knlPntIdxColMap( cntPos, cntPos );

    rm = ( 
          (knlPntIdxRowMap.array() - cntRow).pow(2.0) 
        + (knlPntIdxColMap.array() - cntRow).pow(2.0) 
        ).sqrt().matrix().cast<double>();
}

static void
put_Ws_map( const MatrixXd& distanceMap, double gs, MatrixXd& Ws)
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

    mIndexMapRow  = MatrixXi(windowWidth, windowWidth);
    mIndexMapCol  = MatrixXi(windowWidth, windowWidth);
    mKnlIdxRow    = MatrixXi(windowWidth, windowWidth);
    mKnlIdxCol    = MatrixXi(windowWidth, windowWidth);
    mPntIdxKnlRow = MatrixXi(nw, nw);
    mPntIdxKnlCol = MatrixXi(nw, nw);

    mDistanceMap = MatrixXd(windowWidth, windowWidth);
    mWsMap       = MatrixXd(windowWidth, windowWidth);

    mPntDistKnl  = MatrixXd(nw, nw);

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

void BilateralWindowMatcher::set_gamma_s(real gs)
{
    mGammaS = gs;
}

BilateralWindowMatcher::real 
BilateralWindowMatcher::get_gamma_s(void)
{
    return mGammaS;
}

void BilateralWindowMatcher::set_gamma_c(real gc)
{
    mGammaC = gc;
}

BilateralWindowMatcher::real 
BilateralWindowMatcher::get_gamma_c(void)
{
    return mGammaC;
}
