#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "SLFusion/BilateralWindowMatcher.hpp"
#include "SLFusion/SLCommon.hpp"

namespace slf
{

template<typename _ST, typename _DT, typename _VT>
void BilateralWindowMatcher::block_average_based_on_integral_image(
    const Mat& sint, const Mat& mint, Mat& dst, Mat& vc, int row, int col) const
{
    using namespace cv;

    // Calculate the index of the upper-left pixel.
    const int halfCount = half_count(mWindowWidth);

    // Declare variables inside the loop.
    const int channels = sint.channels();
    const int bs   = mKernelSize;     // block shift.
    const int bsch = bs * channels;   // block shift with channels.
    int blockR0    = row - halfCount; // Upper-left row index of a block.
    const int bC0S = (col - halfCount) * channels; // Upper-left col index of a block in sint.
    const int bC0M = col - halfCount; // Upper-left col index of a block in mint.
    int blockC0S   = 0;
    int blockC0M   = 0;
    int blockR1    = 0;               // blockR1 = blockR0 + bs.
    int blockC1S   = 0;               // blockC1S = blcokC0S + bsch.
    int blockC1M   = 0;               // blockC1M = blcokC0M + bs.
    int bCK0 = 0, bCK1 = 0;           // bCK1 = bCK0 + bs.
    _ST  tempSum   = (_ST)( 0 );
    int  tempNum   = 0;
    _DT* pDst      = NULL;
    int  colDst    = 0;
    _VT* pVc       = NULL;

    // Pointers pointing into sint.
    const _ST* pS0 = NULL; // Upper.
    const _ST* pS1 = NULL; // Lower.

    // Loop.
    for ( int i = 0; i < mNumKernels; ++i )
    {
        pDst   = dst.ptr<_DT>( i );
        pVc    = vc.ptr<_VT>( i );
        colDst = 0; // Reset the column index of dst.
        // Reset the column index of sint and mint.
        blockC0S = bC0S;
        blockC0M = bC0M;
        blockR1  = blockR0 + bs; // Lower row index of sint and mint.
        
        // Get the pointers of sint.
        pS0 = sint.ptr<_ST>( blockR0 );
        pS1 = sint.ptr<_ST>( blockR1 );

        // Loop over columns.
        for ( int j = 0; j < mNumKernels; ++j )
        {
            blockC1M = blockC0M + bs; // Right column index of mint ONLY.

            // Number of valid pixels inside a block/kernel.
            tempNum = 
                  mint.at<int>( blockR1, blockC1M ) 
                - mint.at<int>( blockR0, blockC1M ) 
                - mint.at<int>( blockR1, blockC0M ) 
                + mint.at<int>( blockR0, blockC0M );
            
            pVc[j] = tempNum; // Only one channel.

            if ( 0 != tempNum )
            {
                bCK0 = blockC0S;    // Left column index of sint.
                bCK1 = bCK0 + bsch; // Right column index of sint.

                for ( int k = 0; k < channels; ++k )
                {
                    tempSum = 
                          pS1[ bCK1 ] - pS0[ bCK1 ] - pS1[ bCK0 ] + pS0[ bCK0 ];

                    pDst[ colDst + k ] = (_DT)( tempSum / tempNum ); // Average.
                    
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

            colDst   += channels;
            blockC0S += bsch;
            blockC0M += bs;
        }

        blockR0 += bs; // Grow the upper-left row index.
    }
}

template <typename _ST, typename _IT> 
void BilateralWindowMatcher::match_single_line(
        const Mat& refMat, const Mat& tstMat, 
        const Mat& refInt, const Mat& tstInt,
        const Mat& refMInt, const Mat& tstMInt,
        int rowIdx,
        int minDisp, int maxDisp, 
        MatchingCost<Real_t>* pMC, int* nMC)
{
    using namespace cv;
    using namespace Eigen;

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

    if ( refInt.cols != tstInt.cols )
    {
        std::stringstream ssRef;
        ssRef << "( " << refInt.cols << ", " << refInt.rows << " )";

        std::stringstream ssTst;
        ssTst << "( " << tstInt.cols << ", " << tstInt.rows << " )";

        EXCEPTION_DIMENSION_MISMATCH(refInt, ssRef.str(), tstInt, ssTst.str());
    }

    if ( refMInt.cols != tstMInt.cols )
    {
        std::stringstream ssRef;
        ssRef << "( " << refMInt.cols << ", " << refMInt.rows << " )";

        std::stringstream ssTst;
        ssTst << "( " << tstMInt.cols << ", " << tstMInt.rows << " )";

        EXCEPTION_DIMENSION_MISMATCH(refMInt, ssRef.str(), tstMInt, ssTst.str());
    }

    if ( refMat.cols + 1 != refMInt.cols || refMat.cols + 1 != refInt.cols )
    {
        std::stringstream ssRef;
        ssRef << "( " << refMat.cols << ", " << refMat.rows << " )";

        std::stringstream ssMInt;
        ssMInt << "( " << refMInt.cols << ", " << refMInt.rows << " )";

        EXCEPTION_DIMENSION_MISMATCH(refMat, ssRef.str(), refMInt, ssMInt.str());
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
    Mat vc( mNumKernels, mNumKernels, CV_8UC1 );
    FMatrix_t wc(mNumKernels, mNumKernels);

    Mat ac;
    if ( 3 == refInt.channels() )
    {
        ac = Mat(mNumKernels, mNumKernels, CV_32FC3);
    }
    else
    {
        ac = Mat(mNumKernels, mNumKernels, CV_32FC1);
    }

    for ( int i = 0; i < pixels; ++i )
    {
        if ( true == mFlagDebug && debug_get_size_index_avg_color() > 0 )
        {
            if ( i == debug_get_next_index_avg_color() )
            {
                std::cout << "Debug AC, i = " << i << std::endl;
            }
        }

        // ============================
        // = Calculate weight matrix. =
        // ============================

        // mWCO.wc( windowRef, winMIntRef, wc, ac );
        block_average_based_on_integral_image<_IT, Real_t, uchar>( refInt, refMInt, ac, vc, rowIdx, idxRef );
        mWCO.wc<Real_t>( ac, vc, wc );

        expand_block_2_window_mat<Real_t>( ac, mACArrayRef[i] );
        expand_block_2_window_matrix<Real_t>( wc, mWCArrayRef[i] );

        // mWCO.wc( windowTst, winMIntTst, wc, ac );
        block_average_based_on_integral_image<_IT, Real_t, uchar>( tstInt, tstMInt, ac, vc, rowIdx, idxTst );
        mWCO.wc<Real_t>( ac, vc, wc );

        expand_block_2_window_mat<Real_t>( ac, mACArrayTst[i] );
        expand_block_2_window_matrix<Real_t>( wc, mWCArrayTst[i] );

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

                debug_in_loop_wc_avg_color( i,
                    mACArrayRef[i], mACArrayTst[i],
                    windowRef, windowTst, winMIntRef, winMIntTst,
                    mWCArrayRef[i], mWCArrayTst[i] );

                debug_pop_index_avg_color();
            }
        }
    }

    // FMatrix_t tad( mNumKernels, mNumKernels );
    FMatrix_t tad( mWindowWidth, mWindowWidth );
    int  idxAvgColorArrayTst = 0; // The index for avgColorArrayTst.
    FMatrix_t tempDenominatorMatrix;
    Real_t  tempCost = 0.0;

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
            // TADm<Real_t, Real_t>( mACArrayRef[i], mACArrayTst[idxAvgColorArrayTst], tad );
            TADm<_ST, _ST>( windowRef, windowTst, tad );

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

} // namespace slf.