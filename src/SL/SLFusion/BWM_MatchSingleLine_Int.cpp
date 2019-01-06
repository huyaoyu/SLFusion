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

template<typename _ST, typename _DT>
void BilateralWindowMatcher::block_average_based_on_integral_image(
    const Mat& sint, const Mat& mint, Mat& dst, int row, int col) const
{
    // Calculate the index of the upper-left pixel.
    const int halfCount = half_count(mWindowWidth);
    int idxR = row - halfCount;
    int idxC = col - halfCount;

    // Declare variables inside the loop.
    const int channels = sint.channels();
    int blockR   = 0; // Upper-left row index of a block.
    int blockC   = 0; // Upper-left col index of a block.
    const int bs = mKernelSize - 1; // block shift.
    int blockR1  = 0; // blockR1 = blockR + bs.
    int blockC1  = 0; // blockC1 = blcokC + bs.
    int bRK0 = 0, bRK1 = 0; // bRK1 = bRK0 + bs.
    int bCK0 = 0, bCK1 = 0; // bCK1 = bCK0 + bs.
    _ST  tempSum = (_ST)( 0 );
    int  tempNum = 0;
    _DT* pDst    = NULL;
    int  colDst  = 0;

    // Loop.
    for ( int i = 0; i < mNumKernels; ++i )
    {
        pDst   = dst.ptr<_DT>( i );
        blockC = 0; // Reset the column index.
        colDst = 0; // Reset the column index.

        for ( int j = 0; j < mNumKernels; ++j )
        {
            blockR1 = blockR + bs; // Lower row index.
            blockC1 = blockC + bs; // Right column index.

            tempNum = 
                  mint.at<int>( blockR1, blockC1 ) 
                - mint.at<int>( blockR,  blockC1 ) 
                - mint.at<int>( blockR1, blockC ) 
                + mint.at<int>( blockR,  blockC );

            if ( 0 != tempNum )
            {
                bRK0 = blockR;    // Upper row index.
                bRK1 = bRK0 + bs; // Lower row index.
                bCK0 = blockC;    // Left column index.
                bCK1 = bCK0 + bs; // Right column index.

                for ( int k = 0; k < channels; ++k )
                {    
                    tempSum = 
                          sint.at<_ST>( bRK1, bCK1 ) 
                        - sint.at<_ST>( bRK0, bCK1 ) 
                        - sint.at<_ST>( bRK1, bCK0 ) 
                        + sint.at<_ST>( bRK0, bCK0 );

                    pDst[ colDst + k ] = tempSum / tempNum; // Average.
                    
                    bRK0++;
                    bRK1++;
                    bCK0++;
                    bCK1++;
                }
            }
            else
            {
                for ( int k = 0; k < channels; ++k )
                {
                    pDst[ colDst + k ] = (_DT)( 0 );
                }
            }

            colDst += channels;
            blockC += mKernelSize;
        }

        blockR += mKernelSize; // Set the upper-left row index.
    }
}

template <typename _IT> 
void BilateralWindowMatcher::match_single_line(
        const Mat& refMat, const Mat& tstMat, 
        const Mat& refInt, const Mat& tstInt,
        const Mat& refMInt, const Mat& tstMInt,
        int rowIdx,
        int minDisp, int maxDisp, 
        MatchingCost<Real_t>* pMC, int* nMC)
{
    if ( true == mFlagDebug )
    {
        std::vector<int> jpegParams;
        jpegParams.push_back(IMWRITE_JPEG_QUALITY);
        jpegParams.push_back(100);

        imwrite("refMat.jpg",   refMat, jpegParams);
        imwrite("tstMat.jpg",   tstMat, jpegParams);
        imwrite("refMInt.jpg", refMInt, jpegParams);
        imwrite("tstMInt.jpg", tstMInt, jpegParams);
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

    if ( refMInt.cols != tstMInt.cols )
    {
        std::stringstream ssRef;
        ssRef << "( " << refMInt.cols << ", " << refMInt.rows << " )";

        std::stringstream ssTst;
        ssTst << "( " << tstMInt.cols << ", " << tstMInt.rows << " )";

        EXCEPTION_DIMENSION_MISMATCH(refMInt, ssRef.str(), tstMInt, ssTst.str());
    }

    if ( refMat.cols != refMInt.cols )
    {
        std::stringstream ssRef;
        ssRef << "( " << refMat.cols << ", " << refMat.rows << " )";

        std::stringstream ssMask;
        ssMask << "( " << refMInt.cols << ", " << refMInt.rows << " )";

        EXCEPTION_DIMENSION_MISMATCH(refMat, ssRef.str(), refMInt, ssMask.str());
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
    const Range rowRange( rowIdx - halfCount, rowIdx + halfCount + 1 );
    Range colRangeRef( idxRef - halfCount, idxRef + halfCount + 1 );
    Range colRangeTst( idxTst - halfCount, idxTst + halfCount + 1 );

    Mat windowRef, windowTst;
    Mat winMIntRef, winMIntTst;
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
        winMIntRef = refMInt( rowRange, colRangeRef );
        winMIntTst = tstMInt( rowRange, colRangeTst );

        // Calculate weight matrix.

        // Memory allocation for avgColorArrayRef and avgColorArrayTst will occur inside mWCO.wc().
        mWCO.wc( windowRef, winMIntRef, wc, ac );
        expand_block_2_window_mat<R_t>( ac, mACArrayRef[i] );
        expand_block_2_window_matrix<R_t>( wc, mWCArrayRef[i] );
        mWCO.wc( windowTst, winMIntTst, wc, ac );
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
                    windowRef, windowTst, winMIntRef, winMIntTst,
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

    // Reset the indices.
    idxRef = minDisp + halfCount, idxTst = halfCount;

    // === Calculate the cost. ===
    int debugCount = 0;
    for ( int i = 0; i < pixels; ++i )
    {
        // The index in the original reference image.
        idxRef = mPixelIdxRef[i];
        // Column index in the original reference image.
        colRangeRef.start = idxRef - halfCount;
        colRangeRef.end   = idxRef + halfCount + 1;
        // Take out the window from the reference image.
        windowRef = refMat( rowRange, colRangeRef );

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

            // The index in the original test image.
            idxTst = mPixelIdxTst[idxAvgColorArrayTst];
            // Column index in the original test image.
            colRangeTst.start = idxTst - halfCount;
            colRangeTst.end   = idxTst + halfCount + 1;
            // Take out the window from the test image.
            windowTst = tstMat( rowRange, colRangeTst );

            if ( true == mFlagDebug )
            {
                if ( i == mDebug_ABIdx0 && j == mDebug_ABIdx1 )
                {
                    std::cout << "Debug." << std::endl; // This line is for placing a breakpoint.
                }
            }

            // Calculate the TAD over all the kernel blocks of windowRef and windowTst.
            // TADm<R_t, R_t>( mACArrayRef[i], mACArrayTst[idxAvgColorArrayTst], tad );
            TADm<uchar, uchar>( windowRef, windowTst, tad );

            // Calculate the cost value.
            tempDenominatorMatrix = ( mWss.array() * mWCArrayRef[i].array() * mWCArrayTst[idxAvgColorArrayTst].array() ).matrix();
            // tempDenominatorMatrix = mWss;

            tempCost = 
                ( tempDenominatorMatrix.array() * tad.array() ).sum() / tempDenominatorMatrix.sum();
            // tempCost = tad.array().mean();

            // Save the cost value into pMC.
            pMC[i].push_back( idxRef - mPixelIdxTst[idxAvgColorArrayTst], tempCost );

            if ( true == mFlagDebug )
            {
                if ( i == mDebug_ABIdx0 )
                {
                    debug_in_loop_cost(i, 
                        tempCost, idxRef - mPixelIdxTst[idxAvgColorArrayTst], idxAvgColorArrayTst,
                        windowRef, windowTst, 
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