#include <cmath>

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
  mGammaS(14), mGammaC(23), mTAD_T(100)
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

void BilateralWindowMatcher::put_wc(const Mat& src, FMatrix_t& wc, Mat& avgColor, Mat* bufferS)
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
    put_average_color_values( src, avgColor );

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

    pAvgColorVal = avgColor.ptr<Real_t>( centerIdx );
    for ( int i = 0; i < channels; ++i )
    {
        colorSrc[i] = *( pAvgColorVal + centerIdx*channels + i );
    }

    for ( int i = 0; i < avgColor.rows; ++i )
    {
        pAvgColorVal = avgColor.ptr<Real_t>( i );
        posCol       = 0;
        
        for ( int j = 0; j < avgColor.cols; ++j )
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

void BilateralWindowMatcher::TADm(const Mat& ref, const Mat& tst, FMatrix_t& tad)
{
    // The rows and cols of ref and tst are assumed to be the same.
    const int channels = ref.channels();
    const uchar* pRef  = NULL;
    const uchar* pTst  = NULL;

    R_t temp = 0.0;

    int posCol = 0, posColShift = 0;
    int posTad = 0;

    for ( int i = 0; i < ref.rows; ++i )
    {
        // Get the pointer to the ref and tst.
        pRef = ref.ptr<uchar>(i);
        pTst = ref.ptr<uchar>(i);

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

void BilateralWindowMatcher::match_single_line(
        const Mat& refMat, const Mat& tstMat, int rowIdx,
        int minDisp, int maxDisp, 
        MatchingCost<BilateralWindowMatcher::Real_t>* pMC, int* nMC )
{
    // Expecting input images have the same width.
    if ( refMat.cols != tstMat.cols )
    {
        std::stringstream ssRef;
        ssRef << "( " << refMat.cols << ", " << refMat.rows << " )";

        std::stringstream ssTst;
        ssTst << "( " << tstMat.cols << ", " << tstMat.rows << " )";

        EXCEPTION_DIMENSION_MISMATCH(refMat, ssRef.str(), tstMat, ssTst.str());
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

    const int numDisp = maxDisp - minDisp;
    const int pixels  = refMat.cols - minDisp - halfCount * 2;

    // === Pre-allocation of weights. ===
    const int nC     = pixels * numDisp;
    FM_t* wcArrayRef = new FM_t[pixels];
    FM_t* wcArrayTst = new FM_t[pixels];

    // == Pre-allocation of average color. ===
    Mat* avgColorArrayRef = new Mat[pixels];
    Mat* avgColorArrayTst = new Mat[pixels];

    // === Calculate color weights for all valid pixels. ===

    int idxRef = minDisp + halfCount, idxTst = halfCount;
    Mat windowRef, windowTst;
    Range rowRange( rowIdx - halfCount, rowIdx + halfCount + 1 );

    // Buffers.
    Mat bufferS( mWindowWidth, mWindowWidth, OCV_F_TYPE );

    for ( int i = 0; i < pixels; ++i )
    {
        // Update the ROIs in refMat and tstMat.
        Range colRangeRef( idxRef - halfCount, idxRef + halfCount + 1 );
        Range colRangeTst( idxTst - halfCount, idxTst + halfCount + 1 );

        windowRef = refMat( rowRange, colRangeRef );
        windowTst = tstMat( rowRange, colRangeTst );

        // Allocate memory for the current color weight matrix.
        wcArrayRef[i] = FM_t::Zero( mNumKernels, mNumKernels );
        wcArrayTst[i] = FM_t::Zero( mNumKernels, mNumKernels );

        // Calculate weight matrix.
        // avgColorArrayRef[i].create( mNumKernels, mNumKernels,  );
        // avgColorArrayTst[i].create( mNumKernels, mNumKernels,  );
        // Memory allocation for avgColorArrayRef and avgColorArrayTst will occur inside put_wc().
        put_wc( windowRef, wcArrayRef[i], avgColorArrayRef[i], &bufferS );
        put_wc( windowTst, wcArrayTst[i], avgColorArrayTst[i], &bufferS );

        // Update indices.
        idxRef++;
        idxTst++;
    }

    FM_t tad( mNumKernels, mNumKernels );
    int idxAvgColorArrayTst = 0; // The index for avgColorArrayTst.
    FM_t tempDenominatorMatrix;
    R_t  tempCost = 0.0;

    // === Calculate the cost. ===
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

            // Calculate the TAD over all the kernel blocks of windowRef and windowTst.
            TADm( avgColorArrayRef[i], avgColorArrayTst[idxAvgColorArrayTst], tad );

            // Calculate the cost value.
            tempDenominatorMatrix = ( mWss.array() * wcArrayRef[i].array() * wcArrayTst[idxAvgColorArrayTst].array() ).matrix();

            tempCost = 
                ( tempDenominatorMatrix.array() * tad.array() ).sum() / 
                tempDenominatorMatrix.sum();

            // Save the cost value into pMC.
            pMC[i].push_back( j + 1, tempCost );
        }

        // // Debug.
        // std::cout << "i = " << i << std::endl;
    }

    // Debug.
    std::cout << "Costs calculated." << std::endl;

    // Release resources.
    delete [] avgColorArrayTst; avgColorArrayTst = NULL;
    delete [] avgColorArrayRef; avgColorArrayRef = NULL;

    delete [] wcArrayTst; wcArrayTst = NULL;
    delete [] wcArrayRef; wcArrayRef = NULL;
}
