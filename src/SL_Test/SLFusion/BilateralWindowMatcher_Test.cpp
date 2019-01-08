#include <exception>
#include <iostream>
#include <string>

#include <boost/move/unique_ptr.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "SLFusion/BilateralWindowMatcher.hpp"
#include "SLFusion/SLCommon.hpp"
#include "SLFusion/SLFusion.hpp"

using namespace cv;
using namespace Eigen;
using namespace std;

const int slf::Test_BilateralWindowMatcher::mDefaultKernelSize  = 3;
const int slf::Test_BilateralWindowMatcher::mDefaultNumKernels  = 13;
const int slf::Test_BilateralWindowMatcher::mDefaultWindowWidth = 3 * 13;

namespace slf
{

typedef IMatrix_t IM_t;
typedef FMatrix_t FM_t;
typedef Real_t    R_t;

TEST_F( Test_BilateralWindowMatcher, getter_setter )
{
    // Create an object to work with.
    BilateralWindowMatcher bwm(mDefaultKernelSize, mDefaultNumKernels);

    bwm.set_gamma_c(12.0);
    bwm.set_gamma_s(19.0);

    ASSERT_EQ( bwm.get_kernel_size(), mDefaultKernelSize ) << "Kernel size.";
    ASSERT_EQ( bwm.get_num_kernels_single_side(), mDefaultNumKernels ) << "Number of kernels.";
    ASSERT_EQ( bwm.get_window_width(), mDefaultWindowWidth ) << "Window width.";
    ASSERT_EQ( bwm.get_gamma_c(), 12.0 ) << "gamma_c";
    ASSERT_EQ( bwm.get_gamma_s(), 19.0 ) << "gamma_s";
}

TEST_F( Test_BilateralWindowMatcher, distance_map )
{
    // Create an object to work with.
    BilateralWindowMatcher bwm(mDefaultKernelSize, mDefaultNumKernels);

    int last   = bwm.get_window_width() - 1;
    int center = last / 2; 
    R_t eps    = 1e-5;
    R_t dist0  = 25.45584412271571;

    ASSERT_EQ( bwm.mDistanceMap(center, center), 0 ) << "The center distance is zero.";
    ASSERT_LT( fabs( ( bwm.mDistanceMap(0, 0) - dist0 ) / dist0 ), eps ) << "The upper left corner.";
    ASSERT_LT( fabs( ( bwm.mDistanceMap(last, last) - dist0 ) / dist0 ), eps ) << "The bottom right corner.";

    ASSERT_EQ( bwm.mDistanceMap(0, 0), bwm.mDistanceMap(1, 1) ) << " (0, 0) and (1, 1) have the same distance.";
    ASSERT_EQ( bwm.mDistanceMap(last, last), bwm.mDistanceMap( last - 1, last - 1) ) << " (last, last) and (last - 1, last - 1) have the same distance.";
}

TEST_F( Test_BilateralWindowMatcher, ws )
{
    // Create an object to work with.
    BilateralWindowMatcher bwm(mDefaultKernelSize, mDefaultNumKernels);
    bwm.set_gamma_s( 25 );

    int last   = bwm.get_window_width() - 1;
    int center = last / 2; 
    R_t eps    = 1e-5;
    R_t ws0    = 0.3612323983950063;

    ASSERT_EQ( bwm.mWsMap(center, center), 1 ) << "The center ws is 1.";
    ASSERT_LT( fabs( ( bwm.mWsMap(0, 0) - ws0 ) / ws0 ), eps ) << "The upper left corner.";
    ASSERT_LT( fabs( ( bwm.mWsMap(last, last) - ws0 ) / ws0 ), eps ) << "The bottom right corner.";

    ASSERT_EQ( bwm.mWsMap(0, 0), bwm.mWsMap(1, 1) ) << " (0, 0) and (1, 1) have the same ws.";
    ASSERT_EQ( bwm.mWsMap(last, last), bwm.mWsMap( last - 1, last - 1) ) << " (last, last) and (last - 1, last - 1) have the same ws.";
}

TEST_F( Test_BilateralWindowMatcher, wss )
{
    // Create an object to work with.
    BilateralWindowMatcher bwm(mDefaultKernelSize, mDefaultNumKernels);
    bwm.set_gamma_s( 25 );

    int last   = bwm.get_window_width() - 1;
    int center = last / 2; 
    R_t eps    = 1e-5;
    R_t wss0   = 0.13048884565020857;

    ASSERT_EQ( bwm.mWss(center, center), 1 ) << "The center ws is 1.";
    ASSERT_LT( fabs( ( bwm.mWss(0, 0) - wss0 ) / wss0 ), eps ) << "The upper left corner.";
    ASSERT_LT( fabs( ( bwm.mWss(last, last) - wss0 ) / wss0 ), eps ) << "The bottom right corner.";

    ASSERT_EQ( bwm.mWss(0, 0), bwm.mWss(0, last) ) << " (0, 0) and (0, last) have the same wss.";
    ASSERT_EQ( bwm.mWss(last, 0), bwm.mWss( last, last) ) << " (last, 0) and (last, last) have the same wss.";
}

TEST_F( Test_BilateralWindowMatcher, inner_pixels )
{
    BilateralWindowMatcher bwm( 3, 13 );

    ASSERT_EQ( bwm.num_inner_pixels( 138, 20, 19 ), 80 ) << "The number of inner pixels.";
}

TEST_F( Test_BilateralWindowMatcher, create_array_buffer )
{
    // Create an object to work with.
    BilateralWindowMatcher bwm(mDefaultKernelSize, mDefaultNumKernels);

    size_t size   = 4000;
    int cvMatType = CV_32FC1;

    // Call private member method to create buffers.
    bwm.create_array_buffer( size, cvMatType, false );

    ASSERT_EQ( bwm.mABSize, 4000 ) << "The inner buffer size (count).";
    
    // Save the buffer header pointers for now.
    void* tempACArrayRef = (void*)( bwm.mACArrayRef );
    void* tempACArrayTst = (void*)( bwm.mACArrayTst );
    void* tempWCArrayRef = (void*)( bwm.mWCArrayRef );
    void* tempWCArrayTst = (void*)( bwm.mWCArrayTst );

    // Try to create the buffer with the same size.
    bwm.create_array_buffer( size, cvMatType, false );

    // The buffer header pointers should not be changed.
    ASSERT_EQ( tempACArrayRef, (void*)(bwm.mACArrayRef) ) << "The buffer header pointer mACArrayRef";
    ASSERT_EQ( tempACArrayTst, (void*)(bwm.mACArrayTst) ) << "The buffer header pointer mACArrayTst";
    ASSERT_EQ( tempWCArrayRef, (void*)(bwm.mWCArrayRef) ) << "The buffer header pointer mWCArrayRef";
    ASSERT_EQ( tempWCArrayTst, (void*)(bwm.mWCArrayTst) ) << "The buffer header pointer mWCArrayTst";

    // Try to create the buffer with smaller size without force flag.
    bwm.create_array_buffer( size - 1000, cvMatType, false );

    // The buffer header pointers should not be changed.
    ASSERT_EQ( tempACArrayRef, (void*)(bwm.mACArrayRef) ) << "The buffer header pointer mACArrayRef after creation with the smaller size.";
    ASSERT_EQ( tempACArrayTst, (void*)(bwm.mACArrayTst) ) << "The buffer header pointer mACArrayTst after creation with the smaller size.";
    ASSERT_EQ( tempWCArrayRef, (void*)(bwm.mWCArrayRef) ) << "The buffer header pointer mWCArrayRef after creation with the smaller size.";
    ASSERT_EQ( tempWCArrayTst, (void*)(bwm.mWCArrayTst) ) << "The buffer header pointer mWCArrayTst after creation with the smaller size.";

    // But the mABSize must be changed.
    ASSERT_EQ( bwm.mABSize, size - 1000 ) << "mABSize must be changed.";

    // Try to create a larger buffer.
    bwm.create_array_buffer( size + 1000, cvMatType, false );

    // All of the buffer header pointers should be modified.
    ASSERT_NE( tempACArrayRef, (void*)(bwm.mACArrayRef) ) << "The buffer header pointer mACArrayRef after creation with larger size.";
    ASSERT_NE( tempACArrayTst, (void*)(bwm.mACArrayTst) ) << "The buffer header pointer mACArrayTst after creation with larger size.";
    ASSERT_NE( tempWCArrayRef, (void*)(bwm.mWCArrayRef) ) << "The buffer header pointer mWCArrayRef after creation with larger size.";
    ASSERT_NE( tempWCArrayTst, (void*)(bwm.mWCArrayTst) ) << "The buffer header pointer mWCArrayTst after creation with larger size.";
    ASSERT_EQ( bwm.mABSize, size + 1000 ) << "mABSize must be changed after creation with larger size.";

    // Save the buffer header pointers for now.
    tempACArrayRef = (void*)( bwm.mACArrayRef );
    tempACArrayTst = (void*)( bwm.mACArrayTst );
    tempWCArrayRef = (void*)( bwm.mWCArrayRef );
    tempWCArrayTst = (void*)( bwm.mWCArrayTst );

    // Try forcing creation.
    bwm.create_array_buffer( size + 1000, cvMatType, true );

    // All of the buffer header pointers should be modified.
    // ASSERT_NE( tempACArrayRef, (void*)(bwm.mACArrayRef) ) << "After forcing creation of mACArrayRef without changing the size.";
    // ASSERT_NE( tempACArrayTst, (void*)(bwm.mACArrayTst) ) << "After forcing creation of mACArrayTst without changing the size.";
    // ASSERT_NE( tempWCArrayRef, (void*)(bwm.mWCArrayRef) ) << "After forcing creation of mWCArrayRef without changing the size.";
    // ASSERT_NE( tempWCArrayTst, (void*)(bwm.mWCArrayTst) ) << "After forcing creation of mWCArrayTst without changing the size.";
    ASSERT_EQ( bwm.mABSize, size + 1000 ) << "mABSize remains the same after forcing creation without changing the size.";

    // Save the buffer header pointers for now.
    tempACArrayRef = (void*)( bwm.mACArrayRef );
    tempACArrayTst = (void*)( bwm.mACArrayTst );
    tempWCArrayRef = (void*)( bwm.mWCArrayRef );
    tempWCArrayTst = (void*)( bwm.mWCArrayTst );

    // Try forcing creation with smaller size.
    bwm.create_array_buffer( size, cvMatType, true );

    // All of the buffer header pointers should be modified.
    // ASSERT_NE( tempACArrayRef, (void*)(bwm.mACArrayRef) ) << "After forcing creation of mACArrayRef with smaller size.";
    ASSERT_NE( tempACArrayTst, (void*)(bwm.mACArrayTst) ) << "After forcing creation of mACArrayTst with smaller size.";
    ASSERT_NE( tempWCArrayRef, (void*)(bwm.mWCArrayRef) ) << "After forcing creation of mWCArrayRef with smaller size.";
    ASSERT_NE( tempWCArrayTst, (void*)(bwm.mWCArrayTst) ) << "After forcing creation of mWCArrayTst with smaller size.";
    ASSERT_EQ( bwm.mABSize, size ) << "mABSize remains the same after forcing creation without changing the size.";
}

TEST_F( Test_BilateralWindowMatcher, block_average_based_on_integral_image )
{
    // Read the image file.
    // cout << "Before read image." << endl;
    Mat matTestAvgColorValues = 
        imread("../data/SLFusion/DummyImage_TestAverageColorValues.bmp", IMREAD_COLOR);
    
    if ( matTestAvgColorValues.empty() )
    {
        ASSERT_FALSE(true) << "Read ../data/SLFusion/DummyImage_TestAverageColorValues.bmp failed.";
    }

    if ( matTestAvgColorValues.rows != mDefaultWindowWidth || 
         matTestAvgColorValues.cols != mDefaultWindowWidth )
    {
        stringstream ss;
        ss << "The size of the input file must be the same with mDefaultWindowWidth. "
           << "The size of the input file is " << matTestAvgColorValues.size() << ", " 
           << "mDefaultWindowWidth = " << mDefaultWindowWidth;
        ASSERT_FALSE(true) << ss.str();
    }

    // cout << matTestAvgColorValues << endl;
    // cout << "Image read." << endl;

    // Mask.
    Mat mask( matTestAvgColorValues.size(), CV_8UC1 );
    mask.setTo( Scalar::all( SLF_MASK ) ); // MUST set to be 1.

    Mat matAveragedColorValues( mDefaultNumKernels, mDefaultNumKernels, CV_32FC3 );
    Mat matVc( mDefaultNumKernels, mDefaultNumKernels, CV_8UC1 );

    // Create the integral images of the input image and the mask.
    Mat matTACVInt, maskInt;
    integral(matTestAvgColorValues, matTACVInt, CV_32FC3);
    integral(mask, maskInt, CV_32SC1);
    const int halfCount = half_count(mDefaultWindowWidth);

    BilateralWindowMatcher bwm(mDefaultKernelSize, mDefaultNumKernels);
    bwm.block_average_based_on_integral_image<R_t, R_t, uchar>(
        matTACVInt, maskInt, matAveragedColorValues, matVc,
        halfCount, halfCount
    );

    // cout << matAveragedColorValues << endl;

    int lastIdx = mDefaultNumKernels - 1;

    ASSERT_EQ(  4, matAveragedColorValues.at<Vec3f>(0, 0)[0] );
    ASSERT_EQ( 13, matAveragedColorValues.at<Vec3f>(0, 0)[1] );
    ASSERT_EQ( 22, matAveragedColorValues.at<Vec3f>(0, 0)[2] );

    ASSERT_EQ(  4, matAveragedColorValues.at<Vec3f>(lastIdx, lastIdx)[0] );
    ASSERT_EQ( 13, matAveragedColorValues.at<Vec3f>(lastIdx, lastIdx)[1] );
    ASSERT_EQ( 22, matAveragedColorValues.at<Vec3f>(lastIdx, lastIdx)[2] );
}

TEST_F( Test_BilateralWindowMatcher, block_average_based_on_integral_image_with_mask )
{
    // Read the image file.
    // cout << "Before read image." << endl;
    Mat matTestAvgColorValues = 
        imread("../data/SLFusion/DummyImage_TestAverageColorValues.bmp", IMREAD_COLOR);

    if ( matTestAvgColorValues.empty() )
    {
        ASSERT_FALSE(true) << "Read ../data/SLFusion/DummyImage_TestAverageColorValues.bmp failed.";
    }

    if ( matTestAvgColorValues.rows != mDefaultWindowWidth || 
         matTestAvgColorValues.cols != mDefaultWindowWidth )
    {
        stringstream ss;
        ss << "The size of the input file must be the same with mDefaultWindowWidth. "
           << "The size of the input file is " << matTestAvgColorValues.size() << ", " 
           << "mDefaultWindowWidth = " << mDefaultWindowWidth;
        ASSERT_FALSE(true) << ss.str();
    }

    // cout << "Image read." << endl;

    // The input image looks like this:
    // Channel0   Channel1   Channel2
    //  0, 1, 2   9, 10, 11  18, 19, 20
    //  3, 4, 5  12, 13, 14  21, 22, 23
    //  6, 7, 8  15, 16, 17  24, 25, 26

    // Mask.
    Mat mask( matTestAvgColorValues.size(), CV_8UC1 );
    mask.setTo( Scalar::all( SLF_MASK ) );

    // Set second block to all masked.
    mask( Rect( 3, 0, 3, 3 ) ) = Mat::zeros(3, 3, CV_8UC1);

    // Set third block to partially masked.
    mask.at<uchar>( 0, 6 ) = 0;
    mask.at<uchar>( 1, 7 ) = 0;
    mask.at<uchar>( 2, 8 ) = 0;

    mask.at<uchar>( 3, 9 ) = 0; mask.at<uchar>( 3, 10 ) = 0; mask.at<uchar>( 3, 11 ) = 0;
    mask.at<uchar>( 4, 9 ) = 0;
    mask.at<uchar>( 5, 9 ) = 0;

    // cout << "mask.rows = " << mask.rows << ", mask.cols = " << mask.cols << endl;
    // cout << "mask = " << endl << mask << endl;

    Mat matAveragedColorValues( mDefaultNumKernels, mDefaultNumKernels, CV_32FC3 );
    Mat matVc( mDefaultNumKernels, mDefaultNumKernels, CV_8UC1 );

    // Pre-mask the input image.
    Mat maskInv;
    maskInv = SLF_MASK - mask;
    // cout << "maskInv = " << endl << maskInv << endl;
    matTestAvgColorValues.setTo( Scalar::all(0), maskInv );

    // Create the integral images of the input image and the mask.
    Mat matTACVInt, maskInt;
    integral(matTestAvgColorValues, matTACVInt, CV_32FC3);
    integral(mask, maskInt, CV_32SC1);
    const int halfCount = half_count(mDefaultWindowWidth);

    BilateralWindowMatcher bwm(mDefaultKernelSize, mDefaultNumKernels);
    bwm.block_average_based_on_integral_image<R_t, R_t, uchar>(
        matTACVInt, maskInt, matAveragedColorValues, matVc,
        halfCount, halfCount
    );

    // cout << matAveragedColorValues << endl;

    int lastIdx = mDefaultNumKernels - 1;

    ASSERT_EQ(  4, matAveragedColorValues.at<Vec3f>(0, 0)[0] );
    ASSERT_EQ( 13, matAveragedColorValues.at<Vec3f>(0, 0)[1] );
    ASSERT_EQ( 22, matAveragedColorValues.at<Vec3f>(0, 0)[2] );

    ASSERT_EQ(  0, matAveragedColorValues.at<Vec3f>(0, 1)[0] );
    ASSERT_EQ(  0, matAveragedColorValues.at<Vec3f>(0, 1)[1] );
    ASSERT_EQ(  0, matAveragedColorValues.at<Vec3f>(0, 1)[2] );

    ASSERT_EQ(  4, matAveragedColorValues.at<Vec3f>(0, 2)[0] );
    ASSERT_EQ( 13, matAveragedColorValues.at<Vec3f>(0, 2)[1] );
    ASSERT_EQ( 22, matAveragedColorValues.at<Vec3f>(0, 2)[2] );

    ASSERT_EQ(  6, matAveragedColorValues.at<Vec3f>(1, 3)[0] );
    ASSERT_EQ( 15, matAveragedColorValues.at<Vec3f>(1, 3)[1] );
    ASSERT_EQ( 24, matAveragedColorValues.at<Vec3f>(1, 3)[2] );

    ASSERT_EQ(  4, matAveragedColorValues.at<Vec3f>(lastIdx, lastIdx)[0] );
    ASSERT_EQ( 13, matAveragedColorValues.at<Vec3f>(lastIdx, lastIdx)[1] );
    ASSERT_EQ( 22, matAveragedColorValues.at<Vec3f>(lastIdx, lastIdx)[2] );
}

TEST_F( Test_BilateralWindowMatcher, expand_block_2_window_mat )
{
    const int kernelSize  = mDefaultKernelSize;
    const int numKernels  = mDefaultNumKernels;
    const int windowWidth = kernelSize * numKernels;

    // Create an OpenCV Mat object with dimension of numKernels x numKernels.
    Mat src( numKernels, numKernels, CV_32FC3 );
    const int channels = src.channels();
    const int maxCounts = numKernels * numKernels;
    const int shifts[3] = { 0, maxCounts, maxCounts*2 };

    // Fill the elements of src.
    R_t* pS = NULL;
    int count = 0;
    for ( int i = 0; i < src.rows; ++i )
    {
        pS = src.ptr<R_t>(i);
        for ( int j = 0; j < src.cols; ++j )
        {
            for ( int k = 0; k < channels; ++k )
            {
                pS[ j * channels + k ] = count + shifts[k];
            }

            count++;
        }
    }

    // Test the values of src.
    ASSERT_EQ( src.at<Vec3f>( 0, 0 ), Vec3f( 0, maxCounts, maxCounts*2 ) );
    ASSERT_EQ( src.at<Vec3f>( numKernels - 1, numKernels - 1 ), 
        Vec3f( maxCounts - 1 + shifts[0], maxCounts - 1 + shifts[1], maxCounts - 1 + shifts[2] ) );

    // Expansion.
    Mat dst( windowWidth, windowWidth, CV_32FC3 );
    BilateralWindowMatcher bwm(mDefaultKernelSize, mDefaultNumKernels);

    bwm.expand_block_2_window_mat<R_t>( src, dst );

    // Test the values of dst.
    for( int i = 0; i < kernelSize; ++i )
    {
        for ( int j = 0; j < kernelSize; ++j )
        {
            ASSERT_EQ( dst.at<Vec3f>(i, j), Vec3f( 0, shifts[1], shifts[2] ) );
        }
    }

    for( int i = ( numKernels - 1 ) / 2 * kernelSize; i < ( ( numKernels - 1 ) / 2 + 1 ) * kernelSize; ++i )
    {
        for ( int j = ( numKernels - 1 ) / 2 * kernelSize; j < ( ( numKernels - 1 ) / 2 + 1 ) * kernelSize; ++j )
        {
            ASSERT_EQ( dst.at<Vec3f>(i, j), 
                Vec3f( (maxCounts - 1)/2, (maxCounts - 1)/2 + shifts[1], (maxCounts - 1)/2 + shifts[2] ) );
        }
    }

    for( int i = windowWidth - kernelSize; i < windowWidth; ++i )
    {
        for ( int j = windowWidth - kernelSize; j < windowWidth; ++j )
        {
            ASSERT_EQ( dst.at<Vec3f>(i, j), 
                Vec3f( maxCounts - 1, maxCounts - 1 + shifts[1], maxCounts - 1 + shifts[2] ) );
        }
    }
}

TEST_F( Test_BilateralWindowMatcher, expand_block_2_window_matrix )
{
    const int kernelSize  = mDefaultKernelSize;
    const int numKernels  = mDefaultNumKernels;
    const int windowWidth = kernelSize * numKernels;

    // Create an Eigen Matrix object with dimension of numKernels x numKernels.
    FM_t src( numKernels, numKernels );
    const int maxCounts = numKernels * numKernels;

    // Fill the elements of src.
    R_t* pS = src.data();
    for ( int i = 0; i < src.size(); ++i )
    {
        pS[ i ] = i;
    }

    // Test the values of src.
    ASSERT_EQ( src( 0, 0 ), 0 );
    ASSERT_EQ( src( numKernels - 1, numKernels - 1 ), maxCounts - 1 );

    // Expansion.
    FM_t dst( windowWidth, windowWidth );
    BilateralWindowMatcher bwm(mDefaultKernelSize, mDefaultNumKernels);

    // This is amazing that only one template argument is enough to compile.
    bwm.expand_block_2_window_matrix<R_t>( src, dst );

    // cout << "dst = " << endl << dst << endl;

    // Test the values of dst.
    for( int i = 0; i < kernelSize; ++i )
    {
        for ( int j = 0; j < kernelSize; ++j )
        {
            ASSERT_EQ( dst(i, j), 0 );
        }
    }

    for( int i = ( numKernels - 1 ) / 2 * kernelSize; i < ( ( numKernels - 1 ) / 2 + 1 ) * kernelSize; ++i )
    {
        for ( int j = ( numKernels - 1 ) / 2 * kernelSize; j < ( ( numKernels - 1 ) / 2 + 1 ) * kernelSize; ++j )
        {
            ASSERT_EQ( dst(i, j), (maxCounts - 1)/2 );
        }
    }

    for( int i = windowWidth - kernelSize; i < windowWidth; ++i )
    {
        for ( int j = windowWidth - kernelSize; j < windowWidth; ++j )
        {
            ASSERT_EQ( dst(i, j), maxCounts - 1 );
        }
    }
}

TEST_F( Test_BilateralWindowMatcher, TADm_same_ref_tst )
{
    // Create an object to work with.
    BilateralWindowMatcher bwm(mDefaultKernelSize, mDefaultNumKernels);

    // Create an input windows.
    Mat windowRef( mDefaultNumKernels, mDefaultNumKernels, CV_32FC3 );
    windowRef.setTo( Scalar::all( 255 ) );

    Mat windowTst( windowRef.size(), windowRef.type() );
    windowRef.copyTo( windowTst );

    // Create the tad variable.
    FM_t tad( mDefaultNumKernels, mDefaultNumKernels );

    // Calculate TAD.
    bwm.TADm<R_t, R_t>( windowRef, windowTst, tad );
    stringstream ss;

    for ( int i = 0; i < mDefaultNumKernels; ++i )
    {
        for ( int j = 0; j < mDefaultNumKernels; ++j )
        {
            ss.str(""); ss.clear();
            ss << "( " << i << ", " << j << ").";

            ASSERT_EQ( tad(i, j), 0 ) << ss.str();
        }
    }
}

TEST_F( Test_BilateralWindowMatcher, TADm_same_ref_tst_random )
{
    // Create an object to work with.
    BilateralWindowMatcher bwm(mDefaultKernelSize, mDefaultNumKernels);

    // Create an input windows.
    Mat windowRef( mDefaultNumKernels, mDefaultNumKernels, CV_32FC3 );
    // Randomize windowRef.
    randu( windowRef, Scalar( 0, 0, 0 ), Scalar( 256, 256, 256 ) );

    Mat windowTst( windowRef.size(), windowRef.type() );
    windowRef.copyTo( windowTst );

    // Create the tad variable.
    FM_t tad( mDefaultNumKernels, mDefaultNumKernels );

    // Calculate TAD.
    bwm.TADm<R_t, R_t>( windowRef, windowTst, tad );
    stringstream ss;
    R_t eps = 1e-5;

    for ( int i = 0; i < mDefaultNumKernels; ++i )
    {
        for ( int j = 0; j < mDefaultNumKernels; ++j )
        {
            ss.str(""); ss.clear();
            ss << "( " << i << ", " << j << ").";

            ASSERT_LT( std::fabs( tad(i, j) ), eps ) << ss.str();
        }
    }
}

TEST_F( Test_BilateralWindowMatcher, TADm_manual )
{
    // Create an object to work with.
    BilateralWindowMatcher bwm(mDefaultKernelSize, mDefaultNumKernels);

    // Create an input windows.
    Mat windowRef( mDefaultNumKernels, mDefaultNumKernels, CV_32FC3 );
    windowRef.setTo( Scalar::all( 255 ) );

    Mat windowTst( windowRef.size(), windowRef.type() );
    windowRef.copyTo( windowTst );

    // Modify elements of windowRef.
    windowRef.at<Vec3f>( 0, 0 ) = Vec3f(  50, 100, 150 );
    windowTst.at<Vec3f>( 0, 0 ) = Vec3f( 130,  80, 200 );

    // Create the tad variable.
    FM_t tad( mDefaultNumKernels, mDefaultNumKernels );

    // Calculate TAD.
    bwm.TADm<R_t, R_t>( windowRef, windowTst, tad );
    stringstream ss;
    R_t eps = 1e-5;
    R_t first = sqrt( 80*80 + 20*20 + 50*50 );

    ASSERT_LT( fabs( ( tad(0,0) - first ) / first ), eps ) << "The first element of tad shoudl equal the specified value.";

    for ( int i = 0; i < mDefaultNumKernels; ++i )
    {
        for ( int j = 0; j < mDefaultNumKernels; ++j )
        {
            ss.str(""); ss.clear();
            ss << "( " << i << ", " << j << ").";

            if ( i != 0 && j != 0 )
            {
                ASSERT_LT( std::fabs( tad(i, j) ), eps ) << ss.str();
            }
        }
    }
}

#ifndef OMIT_TESTS

TEST_F(Test_BilateralWindowMatcher, match_single_line_01)
{
    using namespace std;

    BilateralWindowMatcher bwm( 3, 13 );

    // Read 1 image.
    string fn = "../data/SLFusion/L.jpg";
    Mat img0;
    const int kernelSize = 3;
    const int numKernels = 13;

    // Define the disparity range.
    const int minDisparity = 500;
    const int maxDisparity = 999;
    const int numDisparity = maxDisparity - minDisparity + 1;
    int pixels = 0;
    const int halfCount = ( kernelSize * numKernels - 1 ) / 2;

    MatchingCost<R_t>* mcArray = NULL;

    try
    {
        img0 = imread( fn, cv::IMREAD_COLOR );

        cout << fn << " is read." << endl 
             << "img0.rows = " << img0.rows << ", "
             << "img0.cols = " << img0.cols 
             << endl;

        // Create an image based on img0.
        Mat img1( img0.size(), img0.type() );
        img0( Rect( minDisparity, 0, img0.cols - minDisparity, img0.rows) ).copyTo(
            img1( Rect( 0, 0, img0.cols - minDisparity, img0.rows ) )
        );
        
        // Padding.
        Mat padded[2], paddedMask[2];
        Scalar s = Scalar(0, 0, 0);
        if ( 0 != Run_SLFusion::put_padded_mat( img0, kernelSize, numKernels, s, padded[0], paddedMask[0]) )
        {
            ASSERT_FALSE(true);
        }

        if ( 0 != Run_SLFusion::put_padded_mat( img1, kernelSize, numKernels, s, padded[1], paddedMask[1]) )
        {
            ASSERT_FALSE(true);
        }

        cout << "Size of padded: (" << padded[0].rows << ", " << padded[0].cols << ")" << endl;

        pixels = padded[0].cols - minDisparity - ( kernelSize * numKernels - 1 );

        cout << "pixels = " << pixels << endl;

        // Pre-allocations.
        mcArray = new MatchingCost<R_t>[ pixels ];
        for ( int i = 0; i < pixels; ++i )
        {
            mcArray[i].allocate(numDisparity);
        }

        cout << "Estimated memory: " << mcArray[0].estimate_storage() * (pixels) / 1024.0 / 1024 << " MB." << endl;

        // Use this only image as both the reference and test images.
        // Calculate matching cost.
        bwm.match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1], 
            halfCount, minDisparity, maxDisparity, mcArray);

        for ( int i = 0; i < pixels; ++i )
        {
            mcArray[i].reset();
        }

        // bwm.enable_debug();
        bwm.debug_set_array_buffer_idx(19);

        bwm.match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1],
            halfCount, minDisparity, maxDisparity, mcArray);

        bwm.disable_debug();

        cout << "Internal buffer size of BilateralWindowMatcher: " 
             << bwm.get_internal_buffer_szie() / 1024.0 / 1024 << " MB." << endl;

        // Verify the matching cost.
    }
    catch ( exception& exp )
    {
        delete [] mcArray; mcArray = NULL;

        cout << "wat() " << exp.what() << endl;
        ASSERT_FALSE(true);
    }

    int nTst = 0;

    // Output the results.
    string mcDir = "../data/SLFusion/match_single_line_01_cost";

    cout << "Write matching costs to filesystem..." << endl;

    for ( int i = 0; i < pixels; ++i )
    {
        mcArray[i].write(mcDir);
    }

    // Verify the results.
    for ( int i = 0; i < pixels; ++i )
    {
        ASSERT_EQ( mcArray[i].get_idx_ref(), minDisparity + halfCount + i ); 
        ASSERT_EQ( mcArray[i].get_disparity_array()[0], minDisparity );
        nTst = mcArray[i].get_n_test();
        ASSERT_EQ( nTst, ( i + 1 < numDisparity ? i + 1 : numDisparity ) );
        ASSERT_EQ( mcArray[i].get_disparity_array()[nTst - 1], ( i + 1 < numDisparity ? i + minDisparity : maxDisparity ) );


        if ( i >= halfCount && i <= pixels - 1 - halfCount )
        {
            for ( int idxC = 0; idxC < nTst; ++idxC )
            {
                if ( 0 == idxC )
                {
                    if ( mcArray[i].get_p_cost()[idxC] >= 1e-12 )
                    {
                        cout << "i = " << i << ", idxC = " << idxC << ", cost = " << mcArray[i].get_p_cost()[idxC] << endl;
                    }
                    ASSERT_LT( mcArray[i].get_p_cost()[idxC], 1e-12 ); // This means the value should essentially be zero.
                }
                else
                {
                    ASSERT_GT( mcArray[i].get_p_cost()[idxC], 0.0 ); // This means not a perfect match.
                }
            }
        }
    }

    delete [] mcArray; mcArray = NULL;
}

TEST_F(Test_BilateralWindowMatcher, match_single_line_02)
{
    using namespace std;

    BilateralWindowMatcher bwm( 3, 13 );

    // Read 1 image.
    string fn = "../data/SLFusion/L.jpg";
    Mat img0;
    const int kernelSize = 3;
    const int numKernels = 13;

    // Define the disparity range.
    const int minDisparity = 500;
    const int maxDisparity = 999;
    const int numDisparity = maxDisparity - minDisparity + 1;
    int pixels = 0;
    const int halfCount = ( kernelSize * numKernels - 1 ) / 2;

    MatchingCost<R_t>* mcArray = NULL;

    try
    {
        img0 = imread( fn, cv::IMREAD_COLOR );

        cout << fn << " is read." << endl 
             << "img0.rows = " << img0.rows << ", "
             << "img0.cols = " << img0.cols 
             << endl;

        // Create an image based on img0.
        Mat img1( img0.size(), img0.type() );
        img0( Rect( minDisparity, 0, img0.cols - minDisparity, img0.rows) ).copyTo(
            img1( Rect( 0, 0, img0.cols - minDisparity, img0.rows ) )
        );

        img0( Rect( 0, 0, minDisparity, img0.rows ) ).copyTo(
            img1( Rect( img0.cols - minDisparity, 0, minDisparity, img0.rows ) )
        );
        
        // Padding.
        Mat padded[2], paddedMask[2];
        Scalar s = Scalar(0, 0, 0);
        if ( 0 != Run_SLFusion::put_padded_mat( img0, kernelSize, numKernels, s, padded[0], paddedMask[0]) )
        {
            ASSERT_FALSE(true);
        }

        if ( 0 != Run_SLFusion::put_padded_mat( img1, kernelSize, numKernels, s, padded[1], paddedMask[1]) )
        {
            ASSERT_FALSE(true);
        }

        cout << "Size of padded: (" << padded[0].rows << ", " << padded[0].cols << ")" << endl;

        pixels = padded[0].cols - minDisparity - ( kernelSize * numKernels - 1 );

        cout << "pixels = " << pixels << endl;

        // Pre-allocations.
        mcArray = new MatchingCost<R_t>[ pixels ];
        for ( int i = 0; i < pixels; ++i )
        {
            mcArray[i].allocate(numDisparity);
        }

        cout << "Estimated memory: " << mcArray[0].estimate_storage() * (pixels) / 1024.0 / 1024 << " MB." << endl;

        // Use this only image as both the reference and test images.
        // Calculate matching cost.
        bwm.match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1], 
            halfCount, minDisparity, maxDisparity, mcArray);

        for ( int i = 0; i < pixels; ++i )
        {
            mcArray[i].reset();
        }

        bwm.enable_debug();
        // bwm.debug_set_array_buffer_idx( pixels - 1 - halfCount );
        bwm.debug_set_array_buffer_idx( pixels - 1 );

        bwm.match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1],
            halfCount, minDisparity, maxDisparity, mcArray);

        bwm.disable_debug();

        cout << "Internal buffer size of BilateralWindowMatcher: " 
             << bwm.get_internal_buffer_szie() / 1024.0 / 1024 << " MB." << endl;

        // Verify the matching cost.
    }
    catch ( exception& exp )
    {
        delete [] mcArray; mcArray = NULL;

        cout << "wat() " << exp.what() << endl;
        ASSERT_FALSE(true);
    }

    int nTst = 0;

    // Output the results.
    string mcDir = "../data/SLFusion/match_single_line_02_cost";

    cout << "Write matching costs to filesystem..." << endl;

    for ( int i = 0; i < pixels; ++i )
    {
        mcArray[i].write(mcDir);
    }

    // Verify the results.
    for ( int i = 0; i < pixels; ++i )
    {
        ASSERT_EQ( mcArray[i].get_idx_ref(), minDisparity + halfCount + i ); 
        ASSERT_EQ( mcArray[i].get_disparity_array()[0], minDisparity );
        nTst = mcArray[i].get_n_test();
        ASSERT_EQ( nTst, ( i + 1 < numDisparity ? i + 1 : numDisparity ) );
        ASSERT_EQ( mcArray[i].get_disparity_array()[nTst - 1], ( i + 1 < numDisparity ? i + minDisparity : maxDisparity ) );


        for ( int idxC = 0; idxC < nTst; ++idxC )
        {
            if ( 0 == idxC && 
                ( ( i >= halfCount && i <= pixels - 1 - halfCount ) || 
                  ( ( i + kernelSize/2 + 1 )%kernelSize == 0 )) )
            {
                if ( mcArray[i].get_p_cost()[idxC] >= 1e-12 )
                {
                    cout << "i = " << i << ", idxC = " << idxC << ", cost = " << mcArray[i].get_p_cost()[idxC] << endl;
                }
                ASSERT_LT( mcArray[i].get_p_cost()[idxC], 1e-12 ); // This means the value should essentially be zero.
            }
            else
            {
                if ( mcArray[i].get_p_cost()[idxC] <= 0 )
                {
                    cout << "i = " << i << ", idxC = " << idxC << ", cost = " << mcArray[i].get_p_cost()[idxC] << endl;
                }
                ASSERT_GT( mcArray[i].get_p_cost()[idxC], 0.0 ); // This means not a perfect match.
            }
        }
    }

    delete [] mcArray; mcArray = NULL;
}

TEST_F(Test_BilateralWindowMatcher, match_single_line_03)
{
    using namespace std;

    BilateralWindowMatcher bwm( 3, 13 );

    // Read 1 image.
    const string fn0 = "../data/SLFusion/L.jpg";
    const string fn1 = "../data/SLFusion/R.jpg";
    Mat tempImg, img0, img1;

    vector<int> jpegParams;
    jpegParams.push_back(IMWRITE_JPEG_QUALITY);
    jpegParams.push_back(100);

    const int width      = 1500;
    const int height     = 100;
    const int kernelSize = 3;
    const int numKernels = 13;

    // Define the disparity range.
    const int minDisparity = 500;
    const int maxDisparity = 999;
    const int numDisparity = maxDisparity - minDisparity + 1;
    int pixels = 0;

    MatchingCost<R_t>* mcArray = NULL;

    try
    {
        tempImg = imread( fn0, cv::IMREAD_COLOR );
        img0    = tempImg( Rect(0, 0, width, height) ).clone();

        cout << fn0 << " is read." << endl 
             << "img0.rows = " << img0.rows << ", "
             << "img0.cols = " << img0.cols 
             << endl;

        // Read img1.
        tempImg = imread( fn1, cv::IMREAD_COLOR );
        img1    = tempImg( Rect(0, 0, width, height) ).clone();

        // Save img0 and img1.
        imwrite( "../data/SLFusion/match_single_line_03_cost/img0.jpg", img0, jpegParams );
        imwrite( "../data/SLFusion/match_single_line_03_cost/img1.jpg", img1, jpegParams );
        
        // Padding.
        Mat padded[2], paddedMask[2];
        Scalar s = Scalar(0, 0, 0);
        if ( 0 != Run_SLFusion::put_padded_mat( img0, kernelSize, numKernels, s, padded[0], paddedMask[0]) )
        {
            ASSERT_FALSE(true);
        }

        if ( 0 != Run_SLFusion::put_padded_mat( img1, kernelSize, numKernels, s, padded[1], paddedMask[1]) )
        {
            ASSERT_FALSE(true);
        }

        cout << "Size of padded: (" << padded[0].rows << ", " << padded[0].cols << ")" << endl;

        imwrite( "../data/SLFusion/match_single_line_03_cost/padded[0].jpg", padded[0], jpegParams );
        imwrite( "../data/SLFusion/match_single_line_03_cost/padded[1].jpg", padded[1], jpegParams );

        pixels = padded[0].cols - minDisparity - ( kernelSize * numKernels - 1 );

        cout << "pixels = " << pixels << endl;

        // Pre-allocations.
        mcArray = new MatchingCost<R_t>[ pixels ];
        for ( int i = 0; i < pixels; ++i )
        {
            mcArray[i].allocate(numDisparity);
        }

        cout << "Estimated memory: " << mcArray[0].estimate_storage() * (pixels) / 1024.0 / 1024 << " MB." << endl;

        // Use this only image as both the reference and test images.
        // Calculate matching cost.
        bwm.match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1],
            ( kernelSize * numKernels - 1 )/2 + height / 2 - 1, minDisparity, maxDisparity, mcArray);

        for ( int i = 0; i < pixels; ++i )
        {
            mcArray[i].reset();
        }

        // bwm.enable_debug();
        // bwm.debug_set_array_buffer_idx(0);

        bwm.match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1],
            ( kernelSize * numKernels - 1 )/2 + height / 2 - 1, minDisparity, maxDisparity, mcArray);
        cout << "Internal buffer size of BilateralWindowMatcher: " 
             << bwm.get_internal_buffer_szie() / 1024.0 / 1024 << " MB." << endl;

        // Verify the matching cost.
    }
    catch ( exception& exp )
    {
        delete [] mcArray; mcArray = NULL;

        cout << "wat() " << exp.what() << endl;
        ASSERT_FALSE(true);
    }

    const int halfCount = ( kernelSize * numKernels - 1 )/2;
    int nTst = 0;

    // Output the results.
    string mcDir = "../data/SLFusion/match_single_line_03_cost";

    cout << "Write matching costs to filesystem..." << endl;

    for ( int i = 0; i < pixels; ++i )
    {
        mcArray[i].write(mcDir);
    }

    delete [] mcArray; mcArray = NULL;
}

TEST_F(Test_BilateralWindowMatcher, match_single_line_04)
{
    using namespace std;

    BilateralWindowMatcher bwm( 3, 13 );

    // Read 1 image.
    string fn = "../data/SLFusion/L.jpg";
    Mat img0;
    const int kernelSize = 3;
    const int numKernels = 13;

    // Define the disparity range.
    const int minDisparity = 500;
    const int maxDisparity = 999;
    const int numDisparity = maxDisparity - minDisparity + 1;
    const int halfCount    = ( kernelSize * numKernels - 1 ) / 2;
    const int shift        = minDisparity / 2;
    int pixels = 0;

    MatchingCost<R_t>* mcArray = NULL;

    try
    {
        img0 = imread( fn, cv::IMREAD_COLOR );

        cout << fn << " is read." << endl 
             << "img0.rows = " << img0.rows << ", "
             << "img0.cols = " << img0.cols 
             << endl;

        // Create an image based on img0.
        Mat img1( img0.size(), img0.type() );
        img0( Rect( minDisparity + shift, 0, img0.cols - minDisparity - shift, img0.rows) ).copyTo(
            img1( Rect( 0, 0, img0.cols - minDisparity - shift, img0.rows ) )
        );

        img0( Rect( 0, 0, minDisparity + shift, img0.rows ) ).copyTo(
            img1( Rect( img0.cols - minDisparity - shift, 0, minDisparity + shift, img0.rows ) )
        );
        
        cout << "Target disparity = " << minDisparity + shift << endl;

        // Padding.
        Mat padded[2], paddedMask[2];
        Scalar s = Scalar(0, 0, 0);
        if ( 0 != Run_SLFusion::put_padded_mat( img0, kernelSize, numKernels, s, padded[0], paddedMask[0]) )
        {
            ASSERT_FALSE(true);
        }

        if ( 0 != Run_SLFusion::put_padded_mat( img1, kernelSize, numKernels, s, padded[1], paddedMask[1]) )
        {
            ASSERT_FALSE(true);
        }

        cout << "Size of padded: (" << padded[0].rows << ", " << padded[0].cols << ")" << endl;

        pixels = padded[0].cols - minDisparity - ( kernelSize * numKernels - 1 );

        cout << "pixels = " << pixels << endl;

        // Pre-allocations.
        mcArray = new MatchingCost<R_t>[ pixels ];
        for ( int i = 0; i < pixels; ++i )
        {
            mcArray[i].allocate(numDisparity);
        }

        cout << "Estimated memory: " << mcArray[0].estimate_storage() * (pixels) / 1024.0 / 1024 << " MB." << endl;

        // Use this only image as both the reference and test images.
        // Calculate matching cost.
        bwm.match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1], 
            padded[0].rows / 2, minDisparity, maxDisparity, mcArray);

        cout << "Internal buffer size of BilateralWindowMatcher: " 
             << bwm.get_internal_buffer_szie() / 1024.0 / 1024 << " MB." << endl;

        // Verify the matching cost.
    }
    catch ( exception& exp )
    {
        delete [] mcArray; mcArray = NULL;

        cout << "wat() " << exp.what() << endl;
        ASSERT_FALSE(true);
    }

    int nTst = 0;

    // Output the results.
    string mcDir = "../data/SLFusion/match_single_line_04_cost";

    cout << "Write matching costs to filesystem..." << endl;

    for ( int i = 0; i < pixels; ++i )
    {
        mcArray[i].write(mcDir);
    }

    delete [] mcArray; mcArray = NULL;
}

TEST_F(Test_BilateralWindowMatcher, match_single_line_05)
{
    using namespace std;

    BilateralWindowMatcher bwm( 3, 13 );

    // Read 1 image.
    const string fn0 = "../data/SLFusion/Sep27_Pillar/Rectified_L_color.jpg";
    const string fn1 = "../data/SLFusion/Sep27_Pillar/Rectified_R_color.jpg";
    Mat tempImg, img0, img1;

    vector<int> jpegParams { IMWRITE_JPEG_QUALITY, 100 };

    const int kernelSize = 3;
    const int numKernels = 13;

    // Define the disparity range.
    const int minDisparity = 600;
    const int maxDisparity = 727;
    const int numDisparity = maxDisparity - minDisparity + 1;
    int pixels = 0;

    MatchingCost<R_t>* mcArray = NULL;

    try
    {
        img0 = imread( fn0, cv::IMREAD_COLOR );

        cout << fn0 << " is read." << endl 
             << "img0.rows = " << img0.rows << ", "
             << "img0.cols = " << img0.cols 
             << endl;

        // Read img1.
        img1 = imread( fn1, cv::IMREAD_COLOR );

        // Save img0 and img1.
        imwrite( "../data/SLFusion/match_single_line_05_cost/img0.jpg", img0, jpegParams );
        imwrite( "../data/SLFusion/match_single_line_05_cost/img1.jpg", img1, jpegParams );
        
        // Convert to CIELab color space.
        // cvtColor( img0, img0, COLOR_BGR2Lab );
        // cvtColor( img1, img1, COLOR_BGR2Lab );
        cvtColor( img0, img0, COLOR_BGR2GRAY );
        cvtColor( img1, img1, COLOR_BGR2GRAY );

        // Mat lapImg[2];
        // Laplacian( img0, lapImg[0], CV_32FC1 );
        // Laplacian( img1, lapImg[1], CV_32FC1 );

        // convertScaleAbs( lapImg[0], lapImg[0] );
        // convertScaleAbs( lapImg[1], lapImg[1] );

        // lapImg[0].convertTo( img0, CV_8UC1 );
        // lapImg[1].convertTo( img1, CV_8UC1 );

        cout << "img0.channels() = " << img0.channels() << endl;

        // Padding.
        Mat padded[2], paddedMask[2];
        Scalar s = Scalar(0, 0, 0);
        if ( 0 != Run_SLFusion::put_padded_mat( img0, kernelSize, numKernels, s, padded[0], paddedMask[0]) )
        {
            ASSERT_FALSE(true);
        }

        if ( 0 != Run_SLFusion::put_padded_mat( img1, kernelSize, numKernels, s, padded[1], paddedMask[1]) )
        {
            ASSERT_FALSE(true);
        }

        cout << "Size of padded: (" << padded[0].rows << ", " << padded[0].cols << ")" << endl;

        imwrite( "../data/SLFusion/match_single_line_05_cost/padded[0].jpg", padded[0], jpegParams );
        imwrite( "../data/SLFusion/match_single_line_05_cost/padded[1].jpg", padded[1], jpegParams );

        pixels = padded[0].cols - minDisparity - ( kernelSize * numKernels - 1 );

        cout << "pixels = " << pixels << endl;

        // Pre-allocations.
        mcArray = new MatchingCost<R_t>[ pixels ];
        for ( int i = 0; i < pixels; ++i )
        {
            mcArray[i].allocate(numDisparity);
        }

        cout << "Estimated memory: " << mcArray[0].estimate_storage() * (pixels) / 1024.0 / 1024 << " MB." << endl;

        // Calculate matching cost.
        
        bwm.enable_debug();
        bwm.debug_set_out_dir("../data/SLFusion/match_single_line_05_cost");
        bwm.debug_set_array_buffer_idx(2434, 0);
        bwm.debug_push_index_avg_color(2434);
        bwm.debug_push_index_avg_color(2434);

        bwm.match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1],
            padded[0].rows / 2, minDisparity, maxDisparity, mcArray);
        cout << "Internal buffer size of BilateralWindowMatcher: " 
             << bwm.get_internal_buffer_szie() / 1024.0 / 1024 << " MB." << endl;

        bwm.disable_debug();

        // Verify the matching cost.
    }
    catch ( exception& exp )
    {
        delete [] mcArray; mcArray = NULL;

        cout << "wat() " << exp.what() << endl;
        ASSERT_FALSE(true);
    }

    const int halfCount = ( kernelSize * numKernels - 1 )/2;
    int nTst = 0;

    // Output the results.
    string mcDir = "../data/SLFusion/match_single_line_05_cost";

    cout << "Write matching costs to filesystem..." << endl;

    for ( int i = 0; i < pixels; ++i )
    {
        mcArray[i].write(mcDir);
    }

    delete [] mcArray; mcArray = NULL;
}

#endif

template <typename _T> 
static void create_checkboard(int height, int width, int low, int high, int type, OutputArray _dst)
{
    _dst.create(height, width, type);
    Mat dst = _dst.getMat();

    _T sL = low;
    _T sH = high;

    const int channels = dst.channels();
    _T* p      = NULL;
    int colPos = 0;
    int flip   = 0;
    _T s;

    for ( int i = 0; i < dst.rows; ++i )
    {
        p = dst.ptr<_T>( i );
        colPos = 0;

        for ( int j = 0; j < dst.cols; ++j )
        {
            s = 0 == flip ? sL : sH;

            for ( int k = 0; k < channels; ++k )
            {
                *( p + colPos + k ) = s;
            }

            colPos += channels;

            flip = 1 - flip;
        }
    }
}

TEST_F(Test_BilateralWindowMatcher, match_single_line_checkerboard)
{
    using namespace std;

    BilateralWindowMatcher bwm( 3, 13 );

    // Read 1 image.
    Mat img0, img1;

    vector<int> jpegParams { IMWRITE_JPEG_QUALITY, 100 };

    const int kernelSize  = 3;
    const int numKernels  = 13;
    const int windowWidth = kernelSize * numKernels;

    // Define the disparity range.
    const int minDisparity = 1;
    const int maxDisparity = 20;
    const int numDisparity = maxDisparity - minDisparity + 1;
    int pixels = 0;

    MatchingCost<R_t>* mcArray = NULL;

    try
    {
        create_checkboard<uchar>( windowWidth, windowWidth + 2*maxDisparity, 0, 255, CV_8UC3, img0 );

        cout << "img0.rows = " << img0.rows << ", "
             << "img0.cols = " << img0.cols 
             << endl;

        create_checkboard<uchar>( windowWidth, windowWidth + 2*maxDisparity, 0, 255, CV_8UC3, img1 );

        // Save img0 and img1.
        imwrite( "../data/SLFusion/match_single_line_06_cost/img0.jpg", img0, jpegParams );
        imwrite( "../data/SLFusion/match_single_line_06_cost/img1.jpg", img1, jpegParams );
        
        // Convert to CIELab color space.
        // cvtColor( img0, img0, COLOR_BGR2Lab );
        // cvtColor( img1, img1, COLOR_BGR2Lab );

        // Padding.
        Mat padded[2], paddedMask[2];
        Scalar s = Scalar(0, 0, 0);
        if ( 0 != Run_SLFusion::put_padded_mat( img0, kernelSize, numKernels, s, padded[0], paddedMask[0]) )
        {
            ASSERT_FALSE(true);
        }

        if ( 0 != Run_SLFusion::put_padded_mat( img1, kernelSize, numKernels, s, padded[1], paddedMask[1]) )
        {
            ASSERT_FALSE(true);
        }

        cout << "Size of padded: (" << padded[0].rows << ", " << padded[0].cols << ")" << endl;

        imwrite( "../data/SLFusion/match_single_line_06_cost/padded[0].jpg", padded[0], jpegParams );
        imwrite( "../data/SLFusion/match_single_line_06_cost/padded[1].jpg", padded[1], jpegParams );

        pixels = padded[0].cols - minDisparity - ( kernelSize * numKernels - 1 );

        cout << "pixels = " << pixels << endl;

        // Pre-allocations.
        mcArray = new MatchingCost<R_t>[ pixels ];
        for ( int i = 0; i < pixels; ++i )
        {
            mcArray[i].allocate(numDisparity);
        }

        cout << "Estimated memory: " << mcArray[0].estimate_storage() * (pixels) / 1024.0 / 1024 << " MB." << endl;

        // Calculate matching cost.
        
        // bwm.enable_debug();
        // bwm.debug_set_array_buffer_idx(19, 0);

        bwm.match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1],
            ( windowWidth - 1 )/2 + img0.rows / 2, minDisparity, maxDisparity, mcArray);
        cout << "Internal buffer size of BilateralWindowMatcher: " 
             << bwm.get_internal_buffer_szie() / 1024.0 / 1024 << " MB." << endl;

        // bwm.disable_debug();

        // Verify the matching cost.
    }
    catch ( exception& exp )
    {
        delete [] mcArray; mcArray = NULL;

        cout << "wat() " << exp.what() << endl;
        ASSERT_FALSE(true);
    }

    const int halfCount = ( kernelSize * numKernels - 1 )/2;
    int nTst = 0;

    // Output the results.
    string mcDir = "../data/SLFusion/match_single_line_06_cost";

    cout << "Write matching costs to filesystem..." << endl;

    for ( int i = 0; i < pixels; ++i )
    {
        mcArray[i].write(mcDir);
    }

    delete [] mcArray; mcArray = NULL;
}

void Test_BilateralWindowMatcher::create_gradient_image( OutputArray _dst, int height, int width, 
        const vector<int>& b, const vector<int>& g, const vector<int>& r )
{
    _dst.create( height, width, CV_8UC3 );
    Mat dst = _dst.getMat();

    const float stepB = 1.0 * ( b[1] - b[0] ) / ( width - 1 );
    const float stepG = 1.0 * ( g[1] - g[0] ) / ( width - 1 );
    const float stepR = 1.0 * ( r[1] - r[0] ) / ( width - 1 );

    int pixel[3] = {0, 0, 0};

    Mat line( 1, width, CV_8UC3 );

    for ( int i = 0; i < width; ++i )
    {
        pixel[0] = (int)( b[0] + stepB * i );
        pixel[1] = (int)( g[0] + stepG * i );
        pixel[2] = (int)( r[0] + stepR * i );

        line.at<Vec3b>(0, i) = Vec3b( pixel[0], pixel[1], pixel[2] );
    }

    for ( int i = 0; i < height; ++i )
    {
        line.copyTo( dst( Rect( 0, i, width, 1 ) ) );
    }
}

TEST_F( Test_BilateralWindowMatcher, match_single_line_gradient )
{
    // Create a gradient 3-channel image.
    const int kernelSize  = 3;
    const int numKernels  = 13;
    const int windowWidth = kernelSize * numKernels;
    const int width       = windowWidth * 2;
    const int height      = windowWidth;
    const int low         = 1;
    const int high        = 255;

    Mat img1( height, width, CV_8UC3 );
    vector<int> rangeB{ low, high };
    vector<int> rangeG{ low, high };
    vector<int> rangeR{ low, high };

    create_gradient_image(img1, height, width, rangeB, rangeG, rangeR);

    // Save the image to file system.
    vector<int> jpegParams{ IMWRITE_JPEG_QUALITY, 100 };
    imwrite( "./GradientWindow.jpg", img1, jpegParams );

    // img0.
    Mat img0(img1.size(), img1.type());

    const int left = ( width - windowWidth ) / 2;
    cout << "left = " << left << endl;
    img0.setTo( Scalar::all(0) );
    img1( Rect( left, 0, windowWidth, height ) ).copyTo( img0( Rect( left + left, 0, windowWidth, height ) ) );

    imwrite( "./GradientWindow_Block.jpg", img0, jpegParams );

    // Padding.
    Mat padded[2], paddedMask[2], gray0, mask0;
    Scalar s = Scalar(0, 0, 0);

    Run_SLFusion::put_padded_mat( img0, kernelSize, numKernels, s, padded[0], paddedMask[0]);
    Run_SLFusion::put_padded_mat( img1, kernelSize, numKernels, s, padded[1], paddedMask[1]);

    cvtColor(padded[0], gray0, COLOR_BGR2GRAY );
    threshold( gray0, mask0, 0, 255, THRESH_BINARY );

    const int minDisparity = 1;
    const int maxDisparity = 100;
    const int numDisparity = maxDisparity - minDisparity + 1;
    const int pixels       = padded[0].cols - minDisparity - ( kernelSize * numKernels - 1 );

    cout << "pixels = " << pixels << endl;

    // Pre-allocations.
    MatchingCost<R_t>* mcArray = new MatchingCost<R_t>[ pixels ];
    for ( int i = 0; i < pixels; ++i )
    {
        mcArray[i].allocate(numDisparity);
    }

    // Create the matcher.
    BilateralWindowMatcher bwm( kernelSize, numKernels );

    bwm.match_single_line( padded[0], padded[1], mask0, paddedMask[1],
        padded[0].rows / 2, minDisparity, maxDisparity, mcArray);

    string mcDir = "../data/SLFusion/match_single_line_gradient_cost";

    cout << "Write matching costs to filesystem..." << endl;

    for ( int i = 0; i < pixels; ++i )
    {
        mcArray[i].write(mcDir);
    }

    delete [] mcArray; mcArray = NULL;
}

TEST_F( Test_BilateralWindowMatcher, match_single_line_mb_tsukuba )
{
    using namespace std;

    BilateralWindowMatcher bwm( 3, 13 );

    // Read 1 image.
    Mat img0, img1;

    vector<int> jpegParams { IMWRITE_JPEG_QUALITY, 100 };

    const int kernelSize  = 3;
    const int numKernels  = 13;
    const int windowWidth = kernelSize * numKernels;

    // Define the disparity range.
    const int minDisparity = 1;
    const int maxDisparity = 32;
    const int numDisparity = maxDisparity - minDisparity + 1;
    int pixels = 0;

    MatchingCost<R_t>* mcArray = NULL;

    try
    {
        img0 = imread( "../data/SLFusion/match_single_line_mb_tsukuba/left.ppm",  IMREAD_COLOR );
        img1 = imread( "../data/SLFusion/match_single_line_mb_tsukuba/right.ppm", IMREAD_COLOR );

        cout << "img0.rows = " << img0.rows << ", "
             << "img0.cols = " << img0.cols 
             << endl;
        
        // Convert to CIELab color space.
        cvtColor( img0, img0, COLOR_BGR2Lab );
        cvtColor( img1, img1, COLOR_BGR2Lab );

        cout << "img0.type() = " << img0.type() << endl;

        // Padding.
        Mat padded[2], paddedMask[2];
        Scalar s = Scalar(0, 0, 0);
        if ( 0 != Run_SLFusion::put_padded_mat( img0, kernelSize, numKernels, s, padded[0], paddedMask[0]) )
        {
            ASSERT_FALSE(true);
        }

        if ( 0 != Run_SLFusion::put_padded_mat( img1, kernelSize, numKernels, s, padded[1], paddedMask[1]) )
        {
            ASSERT_FALSE(true);
        }

        cout << "Size of padded: (" << padded[0].rows << ", " << padded[0].cols << ")" << endl;

        imwrite( "../data/SLFusion/match_single_line_06_cost/padded_0.jpg", padded[0], jpegParams );
        imwrite( "../data/SLFusion/match_single_line_06_cost/padded_1.jpg", padded[1], jpegParams );

        pixels = padded[0].cols - minDisparity - ( kernelSize * numKernels - 1 );

        cout << "pixels = " << pixels << endl;

        // Pre-allocations.
        mcArray = new MatchingCost<R_t>[ pixels ];
        for ( int i = 0; i < pixels; ++i )
        {
            mcArray[i].allocate(numDisparity);
        }

        cout << "Estimated memory: " << mcArray[0].estimate_storage() * (pixels) / 1024.0 / 1024 << " MB." << endl;

        // Calculate matching cost.
        
        bwm.enable_debug();
        bwm.debug_set_out_dir("../data/SLFusion/match_single_line_mb_tsukuba");
        bwm.debug_set_array_buffer_idx(134, 0);
        bwm.debug_push_index_avg_color(134);
        bwm.debug_push_index_avg_color(103);

        bwm.match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1],
            padded[0].rows / 2, minDisparity, maxDisparity, mcArray);
        cout << "Internal buffer size of BilateralWindowMatcher: " 
             << bwm.get_internal_buffer_szie() / 1024.0 / 1024 << " MB." << endl;

        bwm.disable_debug();

        // Verify the matching cost.
    }
    catch ( exception& exp )
    {
        delete [] mcArray; mcArray = NULL;

        cout << "wat() " << exp.what() << endl;
        ASSERT_FALSE(true);
    }

    const int halfCount = ( kernelSize * numKernels - 1 )/2;
    int nTst = 0;

    // Output the results.
    string mcDir = "../data/SLFusion/match_single_line_mb_tsukuba";

    cout << "Write matching costs to filesystem..." << endl;

    for ( int i = 0; i < pixels; ++i )
    {
        mcArray[i].write(mcDir);
    }

    delete [] mcArray; mcArray = NULL;
}

TEST_F( Test_BilateralWindowMatcher, match_single_line_checkerboard_integral_image )
{
using namespace std;

    BilateralWindowMatcher bwm( 3, 13 );

    // Read 1 image.
    Mat img0, img1;

    vector<int> jpegParams { IMWRITE_JPEG_QUALITY, 100 };

    const int kernelSize  = 3;
    const int numKernels  = 13;
    const int windowWidth = kernelSize * numKernels;

    // Define the disparity range.
    const int minDisparity = 1;
    const int maxDisparity = 20;
    const int numDisparity = maxDisparity - minDisparity + 1;

    try
    {
        create_checkboard<uchar>( windowWidth, windowWidth + 2*maxDisparity, 0, 255, CV_8UC3, img0 );

        cout << "img0.rows = " << img0.rows << ", "
             << "img0.cols = " << img0.cols 
             << endl;

        create_checkboard<uchar>( windowWidth, windowWidth + 2*maxDisparity, 0, 255, CV_8UC3, img1 );

        // Save img0 and img1.
        imwrite( "../data/SLFusion/match_single_line_checkerboard_integral_image_cost/img0.jpg", img0, jpegParams );
        imwrite( "../data/SLFusion/match_single_line_checkerboard_integral_image_cost/img1.jpg", img1, jpegParams );
        
        // Convert to CIELab color space.
        // cvtColor( img0, img0, COLOR_BGR2Lab );
        // cvtColor( img1, img1, COLOR_BGR2Lab );

        // Padding.
        Mat padded[2], paddedMask[2];
        Scalar s = Scalar(0, 0, 0);
        if ( 0 != Run_SLFusion::put_padded_mat( img0, kernelSize, numKernels, s, padded[0], paddedMask[0], SLF_MASK) )
        {
            ASSERT_FALSE(true);
        }

        if ( 0 != Run_SLFusion::put_padded_mat( img1, kernelSize, numKernels, s, padded[1], paddedMask[1], SLF_MASK) )
        {
            ASSERT_FALSE(true);
        }

        cout << "Size of padded: (" << padded[0].rows << ", " << padded[0].cols << ")" << endl;

        imwrite( "../data/SLFusion/match_single_line_checkerboard_integral_image_cost/padded[0].jpg", padded[0], jpegParams );
        imwrite( "../data/SLFusion/match_single_line_checkerboard_integral_image_cost/padded[1].jpg", padded[1], jpegParams );

        const int pixels = padded[0].cols - minDisparity - ( kernelSize * numKernels - 1 );

        cout << "pixels = " << pixels << endl;

        // Pre-allocations.
        boost::movelib::unique_ptr<MatchingCost<R_t>[]> mcArray( new MatchingCost<R_t>[ pixels ] );
        for ( int i = 0; i < pixels; ++i )
        {
            mcArray[i].allocate(numDisparity);
        }

        cout << "Estimated memory: " << mcArray[0].estimate_storage() * (pixels) / 1024.0 / 1024 << " MB." << endl;

        // Calculate matching cost.
        
        // bwm.enable_debug();
        // bwm.debug_set_array_buffer_idx(19, 0);

        Mat refInt, tstInt, refMInt, tstMInt;
        Mat maskInv;

        maskInv = SLF_MASK - paddedMask[0];
        padded[0].setTo( Scalar::all(0), maskInv );

        integral( padded[0], refInt, CV_32FC3 );
        integral( paddedMask[0], refMInt, CV_32SC1 );

        maskInv = SLF_MASK - paddedMask[1];
        padded[1].setTo( Scalar::all(0), maskInv );

        integral( padded[1], tstInt, CV_32FC3 );
        integral( paddedMask[1], tstMInt, CV_32SC1 );

        bwm.match_single_line<R_t>( padded[0], padded[1], refInt, tstInt, refMInt, tstMInt, 
            ( windowWidth - 1 )/2 + img0.rows / 2, minDisparity, maxDisparity, mcArray.get() );

        // bwm.match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1],
        //     ( windowWidth - 1 )/2 + img0.rows / 2, minDisparity, maxDisparity, mcArray.get());

        cout << "Internal buffer size of BilateralWindowMatcher: " 
             << bwm.get_internal_buffer_szie() / 1024.0 / 1024 << " MB." << endl;

        // bwm.disable_debug();

        // Verify the matching cost.
        const int halfCount = ( kernelSize * numKernels - 1 )/2;
        int nTst = 0;

        // Output the results.
        string mcDir = "../data/SLFusion/match_single_line_checkerboard_integral_image_cost";

        cout << "Write matching costs to filesystem..." << endl;

        for ( int i = 0; i < pixels; ++i )
        {
            mcArray[i].write(mcDir);
        }
    }
    catch ( exception& exp )
    {
        cout << "wat() " << exp.what() << endl;
        ASSERT_FALSE(true);
    }
}

TEST_F( Test_BilateralWindowMatcher, match_single_line_gradient_integral_image )
{
    // Create a gradient 3-channel image.
    const int kernelSize  = 3;
    const int numKernels  = 13;
    const int windowWidth = kernelSize * numKernels;
    const int width       = windowWidth * 2;
    const int height      = windowWidth;
    const int low         = 1;
    const int high        = 255;

    Mat img1( height, width, CV_8UC3 );
    vector<int> rangeB{ low, high };
    vector<int> rangeG{ low, high };
    vector<int> rangeR{ low, high };

    create_gradient_image(img1, height, width, rangeB, rangeG, rangeR);

    // Save the image to file system.
    vector<int> jpegParams{ IMWRITE_JPEG_QUALITY, 100 };
    imwrite( "../data/SLFusion/match_single_line_gradient_cost_integral_image/GradientWindow.jpg", img1, jpegParams );

    // img0.
    Mat img0(img1.size(), img1.type());

    const int left = ( width - windowWidth ) / 2;
    cout << "left = " << left << endl;
    img0.setTo( Scalar::all(0) );
    img1( Rect( left, 0, windowWidth, height ) ).copyTo( img0( Rect( left + left, 0, windowWidth, height ) ) );

    imwrite( "../data/SLFusion/match_single_line_gradient_cost_integral_image/GradientWindow_Block.jpg", img0, jpegParams );

    // Padding.
    Mat padded[2], paddedMask[2], gray0, mask0;
    Scalar s = Scalar(0, 0, 0);

    Run_SLFusion::put_padded_mat( img0, kernelSize, numKernels, s, padded[0], paddedMask[0], SLF_MASK);
    Run_SLFusion::put_padded_mat( img1, kernelSize, numKernels, s, padded[1], paddedMask[1], SLF_MASK);

    cvtColor(padded[0], gray0, COLOR_BGR2GRAY );
    threshold( gray0, mask0, 0, SLF_MASK, THRESH_BINARY );

    const int minDisparity = 1;
    const int maxDisparity = 100;
    const int numDisparity = maxDisparity - minDisparity + 1;
    const int pixels       = padded[0].cols - minDisparity - ( kernelSize * numKernels - 1 );

    cout << "pixels = " << pixels << endl;

    // Pre-allocations.
    boost::movelib::unique_ptr< MatchingCost<R_t>[] > mcArray( new MatchingCost<R_t>[ pixels ] );
    for ( int i = 0; i < pixels; ++i )
    {
        mcArray[i].allocate(numDisparity);
    }

    Mat refInt, tstInt, refMInt, tstMInt;
    Mat maskInv;

    maskInv = SLF_MASK - paddedMask[0];
    padded[0].setTo( Scalar::all(0), maskInv );

    integral( padded[0], refInt, CV_32FC3 );
    integral( paddedMask[0], refMInt, CV_32SC1 );

    maskInv = SLF_MASK - paddedMask[1];
    padded[1].setTo( Scalar::all(0), maskInv );

    integral( padded[1], tstInt, CV_32FC3 );
    integral( paddedMask[1], tstMInt, CV_32SC1 );

    // Create the matcher.
    BilateralWindowMatcher bwm( kernelSize, numKernels );

    bwm.match_single_line<R_t>( padded[0], padded[1], refInt, tstInt, refMInt, tstMInt, 
        ( windowWidth - 1 )/2 + img0.rows / 2, minDisparity, maxDisparity, mcArray.get() );

    // bwm.match_single_line( padded[0], padded[1], mask0, paddedMask[1],
    //     padded[0].rows / 2, minDisparity, maxDisparity, mcArray);

    string mcDir = "../data/SLFusion/match_single_line_gradient_cost_integral_image";

    cout << "Write matching costs to filesystem..." << endl;

    for ( int i = 0; i < pixels; ++i )
    {
        mcArray[i].write(mcDir);
    }

    // delete [] mcArray; mcArray = NULL;
}

TEST_F( Test_BilateralWindowMatcher, match_single_line_05_integral_image )
{
    using namespace std;

    // Read 1 image.
    const string fn0 = "../data/SLFusion/Sep27_Pillar/Rectified_L_color.jpg";
    const string fn1 = "../data/SLFusion/Sep27_Pillar/Rectified_R_color.jpg";
    Mat tempImg, img0, img1;

    vector<int> jpegParams { IMWRITE_JPEG_QUALITY, 100 };

    const int kernelSize = 3;
    const int numKernels = 13;

    // Define the disparity range.
    const int minDisparity = 600;
    const int maxDisparity = 727;
    const int numDisparity = maxDisparity - minDisparity + 1;
    int pixels = 0;

    try
    {
        img0 = imread( fn0, cv::IMREAD_COLOR );

        cout << fn0 << " is read." << endl 
             << "img0.rows = " << img0.rows << ", "
             << "img0.cols = " << img0.cols 
             << endl;

        // Read img1.
        img1 = imread( fn1, cv::IMREAD_COLOR );

        // Save img0 and img1.
        imwrite( "../data/SLFusion/match_single_line_05_integral_image/img0.jpg", img0, jpegParams );
        imwrite( "../data/SLFusion/match_single_line_05_integral_image/img1.jpg", img1, jpegParams );
        
        // Convert to CIELab color space.
        // cvtColor( img0, img0, COLOR_BGR2Lab );
        // cvtColor( img1, img1, COLOR_BGR2Lab );
        // cvtColor( img0, img0, COLOR_BGR2GRAY );
        // cvtColor( img1, img1, COLOR_BGR2GRAY );

        // Mat lapImg[2];
        // Laplacian( img0, lapImg[0], CV_32FC1 );
        // Laplacian( img1, lapImg[1], CV_32FC1 );

        // convertScaleAbs( lapImg[0], lapImg[0] );
        // convertScaleAbs( lapImg[1], lapImg[1] );

        // lapImg[0].convertTo( img0, CV_8UC1 );
        // lapImg[1].convertTo( img1, CV_8UC1 );

        cout << "img0.channels() = " << img0.channels() << endl;

        // Padding.
        Mat padded[2], paddedMask[2];
        Scalar s = Scalar(0, 0, 0);
        if ( 0 != Run_SLFusion::put_padded_mat( img0, kernelSize, numKernels, s, padded[0], paddedMask[0], SLF_MASK ) )
        {
            ASSERT_FALSE(true);
        }

        if ( 0 != Run_SLFusion::put_padded_mat( img1, kernelSize, numKernels, s, padded[1], paddedMask[1], SLF_MASK ) )
        {
            ASSERT_FALSE(true);
        }

        cout << "Size of padded: (" << padded[0].rows << ", " << padded[0].cols << ")" << endl;

        // imwrite( "../data/SLFusion/match_single_line_05_integral_image/padded[0].jpg", padded[0], jpegParams );
        // imwrite( "../data/SLFusion/match_single_line_05_integral_image/padded[1].jpg", padded[1], jpegParams );

        pixels = padded[0].cols - minDisparity - ( kernelSize * numKernels - 1 );

        cout << "pixels = " << pixels << endl;

        // Pre-allocations.
        boost::movelib::unique_ptr<MatchingCost<R_t>[]> mcArray( new MatchingCost<R_t>[ pixels ] );
        for ( int i = 0; i < pixels; ++i )
        {
            mcArray[i].allocate(numDisparity);
        }

        cout << "Estimated memory: " << mcArray[0].estimate_storage() * (pixels) / 1024.0 / 1024 << " MB." << endl;

        // Calculate matching cost.
        
        Mat refInt, tstInt, refMInt, tstMInt;
        Mat maskInv;

        maskInv = SLF_MASK - paddedMask[0];
        padded[0].setTo( Scalar::all(0), maskInv );

        integral( padded[0], refInt, CV_32FC3 );
        integral( paddedMask[0], refMInt, CV_32SC1 );

        maskInv = SLF_MASK - paddedMask[1];
        padded[1].setTo( Scalar::all(0), maskInv );

        integral( padded[1], tstInt, CV_32FC3 );
        integral( paddedMask[1], tstMInt, CV_32SC1 );

        imwrite( "../data/SLFusion/match_single_line_05_integral_image/padded[0].jpg", padded[0], jpegParams );
        imwrite( "../data/SLFusion/match_single_line_05_integral_image/padded[1].jpg", padded[1], jpegParams );

        BilateralWindowMatcher bwm( kernelSize, numKernels );

        bwm.enable_debug();
        bwm.debug_set_out_dir("../data/SLFusion/match_single_line_05_integral_image");
        bwm.debug_set_array_buffer_idx(2434, 0);
        bwm.debug_push_index_avg_color(2434);
        bwm.debug_push_index_avg_color(2434);

        bwm.match_single_line<R_t>( padded[0], padded[1], refInt, tstInt, refMInt, tstMInt, 
            padded[0].rows / 2, minDisparity, maxDisparity, mcArray.get() );

        // bwm.match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1],
        //     padded[0].rows / 2, minDisparity, maxDisparity, mcArray.get());
        cout << "Internal buffer size of BilateralWindowMatcher: " 
             << bwm.get_internal_buffer_szie() / 1024.0 / 1024 << " MB." << endl;

        bwm.disable_debug();

        // Verify the matching cost.

        const int halfCount = ( kernelSize * numKernels - 1 )/2;
        int nTst = 0;

        // Output the results.
        string mcDir = "../data/SLFusion/match_single_line_05_integral_image";

        cout << "Write matching costs to filesystem..." << endl;

        for ( int i = 0; i < pixels; ++i )
        {
            mcArray[i].write(mcDir);
        }
    }
    catch ( exception& exp )
    {
        // delete [] mcArray; mcArray = NULL;

        cout << "wat() " << exp.what() << endl;
        ASSERT_FALSE(true);
    }

    // delete [] mcArray; mcArray = NULL;
}

}