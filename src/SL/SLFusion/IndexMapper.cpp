#include <cmath>
#include <fstream>
#include <vector>

#include "SLFusion/BilateralWindowMatcher.hpp"

using namespace cv;
using namespace Eigen;
using namespace slf;

typedef IMatrix_t IM_t;
typedef FMatrix_t FM_t;

IndexMapper::IndexMapper(int w, int nw)
{
    int windowWidth = nw * w;

    mIndexMapRow  = IM_t(windowWidth, windowWidth);
    mIndexMapCol  = IM_t(windowWidth, windowWidth);
    mKnlIdxRow    = IM_t(windowWidth, windowWidth);
    mKnlIdxCol    = IM_t(windowWidth, windowWidth);
    mPntIdxKnlRow = IM_t(nw, nw);
    mPntIdxKnlCol = IM_t(nw, nw);

    // Put index maps.
    put_index_map( mIndexMapRow, mIndexMapCol, mKnlIdxRow, mKnlIdxCol, mPntIdxKnlRow, mPntIdxKnlCol, w );
}

IndexMapper::~IndexMapper()
{

}

void IndexMapper::put_index_map(IM_t& ri, IM_t& ci, 
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

void IndexMapper::show(void)
{
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
}
