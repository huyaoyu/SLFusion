#include <exception>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "SLFusion/BilateralWindowMatcher.hpp"
#include "SLFusion/SLFusion.hpp"

using namespace cv;
using namespace Eigen;
using namespace std;

namespace slf
{

typedef BilateralWindowMatcher::IMatrix_t IM_t;
typedef BilateralWindowMatcher::FMatrix_t FM_t;
typedef BilateralWindowMatcher::Real_t    R_t;

BilateralWindowMatcher* Test_BilateralWindowMatcher::mBWM = NULL;

TEST_F( Test_BilateralWindowMatcher, average_color_values )
{
    // Read the image file.
    // cout << "Before read image." << endl;
    Mat matTestAvgColorValues = 
        imread("/home/yaoyu/SourceCodes/SLFusion/data/SLFusion/DummyImage_TestAverageColorValues.bmp", IMREAD_COLOR);
    // cout << "Image read." << endl;

    // Mask.
    Mat mask( matTestAvgColorValues.size(), CV_8UC1 );
    mask.setTo( Scalar::all( 1 ) );

    Mat vcMat, tvcMat;

    Mat matAveragedColorValues;

    // cout << "matTestAvgColorValues.size() = " << matTestAvgColorValues.size() << endl;

    mBWM->put_average_color_values( matTestAvgColorValues, matAveragedColorValues, mask, vcMat, tvcMat );

    // cout << matAveragedColorValues << endl;

    int lastIdx = mBWM->get_num_kernels_single_side() - 1;

    ASSERT_EQ(  4, matAveragedColorValues.at<Vec3f>(0, 0)[0] );
    ASSERT_EQ( 13, matAveragedColorValues.at<Vec3f>(0, 0)[1] );
    ASSERT_EQ( 22, matAveragedColorValues.at<Vec3f>(0, 0)[2] );

    ASSERT_EQ(  4, matAveragedColorValues.at<Vec3f>(lastIdx, lastIdx)[0] );
    ASSERT_EQ( 13, matAveragedColorValues.at<Vec3f>(lastIdx, lastIdx)[1] );
    ASSERT_EQ( 22, matAveragedColorValues.at<Vec3f>(lastIdx, lastIdx)[2] );
}

TEST_F( Test_BilateralWindowMatcher, put_wc_01 )
{
    // Create an input Mat object with all its pixels equal Scalar::all(255).
    const int numKernels  = mBWM->get_num_kernels_single_side();
    const int windowWidth = mBWM->get_window_width();

    Mat src( windowWidth, windowWidth, CV_8UC3 );
    src.setTo( Scalar::all( 255 ) );

    Mat mask( windowWidth, windowWidth, CV_8UC1 );
    mask.setTo( Scalar::all( 1 ) );

    Mat vcMat, tvcMat;

    const int centerIdx = ( numKernels - 1 ) / 2;

    // Resulting matrix.
    FM_t wc = FM_t( numKernels, numKernels );

    Mat bufferK( numKernels, numKernels, mBWM->OCV_F_TYPE );

    // Get the wc values.
    mBWM->put_wc( src, mask, wc, bufferK, vcMat, tvcMat, NULL );

    // cout << "wc = " << endl << wc << endl;

    ASSERT_EQ( wc( 0, 0), 1 );
    ASSERT_EQ( wc( centerIdx, centerIdx), 1 );
    ASSERT_EQ( wc( numKernels - 1, numKernels - 1), 1 );
}

TEST_F( Test_BilateralWindowMatcher, put_wc_02 )
{
    // Create an input Mat object with all its pixels equal Scalar::all(255) except the center one.
    const int numKernels  = mBWM->get_num_kernels_single_side();
    const int windowWidth = mBWM->get_window_width();

    Mat src( windowWidth, windowWidth, CV_8UC3 );
    src.setTo( Scalar::all( 255 ) );

    Mat mask( src.size(), CV_8UC1 );
    mask.setTo( Scalar::all( 1 ) );

    Mat vcMat, tvcMat;

    const int centerIdxSrc = ( windowWidth - 1 ) / 2;
    const int centerIdx    = ( numKernels - 1 ) / 2;

    ASSERT_EQ( centerIdxSrc, 19 );

    src.at<Vec3b>( centerIdxSrc, centerIdxSrc ) = Vec3b( 1, 1, 1 );

    // Resulting matrix.
    FM_t wc = FM_t( numKernels, numKernels );

    Mat bufferK( numKernels, numKernels, mBWM->OCV_F_TYPE );

    // Get the wc values.
    mBWM->put_wc( src, mask, wc, bufferK, vcMat, tvcMat, NULL );

    // cout << "bufferK = " << endl << bufferK << endl;

    // cout << "wc = " << endl << wc << endl;

    const R_t nonCenterValue = 0.11939494898409;
    const R_t eps = 1e-5;

    ASSERT_LT( std::fabs( ( wc(0, 0) - nonCenterValue ) / nonCenterValue ), eps );

    src.at<Vec3b>( centerIdxSrc - 1, centerIdxSrc - 1 ) = Vec3b( 1, 1, 1 );
    src.at<Vec3b>( centerIdxSrc - 1, centerIdxSrc - 0 ) = Vec3b( 1, 1, 1 );
    src.at<Vec3b>( centerIdxSrc - 1, centerIdxSrc + 1 ) = Vec3b( 1, 1, 1 );
    src.at<Vec3b>( centerIdxSrc - 0, centerIdxSrc - 1 ) = Vec3b( 1, 1, 1 );
    src.at<Vec3b>( centerIdxSrc - 0, centerIdxSrc + 1 ) = Vec3b( 1, 1, 1 );
    src.at<Vec3b>( centerIdxSrc + 1, centerIdxSrc - 1 ) = Vec3b( 1, 1, 1 );
    src.at<Vec3b>( centerIdxSrc + 1, centerIdxSrc - 0 ) = Vec3b( 1, 1, 1 );
    src.at<Vec3b>( centerIdxSrc + 1, centerIdxSrc + 1 ) = Vec3b( 1, 1, 1 );

    Mat bufferS( windowWidth, windowWidth, mBWM->OCV_F_TYPE );

    mBWM->put_wc( src, mask, wc, bufferK, vcMat, tvcMat, &bufferS );

    // cout << "wc = " << endl << wc << endl;

    const R_t nonCenterValue2 = 4.930302754597667e-09;

    ASSERT_EQ( wc( centerIdx,     centerIdx     ), 1 );
    ASSERT_LT( std::fabs( ( wc(centerIdx - 1, centerIdx + 1) - nonCenterValue2 ) / nonCenterValue2 ), eps );
    ASSERT_LT( std::fabs( ( wc(centerIdx + 1, centerIdx - 1) - nonCenterValue2 ) / nonCenterValue2 ), eps );
}

#ifndef OMIT_TESTS

TEST_F(Test_BilateralWindowMatcher, match_single_line_01)
{
    using namespace std;

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
        mBWM->match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1], 
            halfCount, minDisparity, maxDisparity, mcArray);

        for ( int i = 0; i < pixels; ++i )
        {
            mcArray[i].reset();
        }

        // mBWM->enable_debug();
        mBWM->debug_set_array_buffer_idx(19);

        mBWM->match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1],
            halfCount, minDisparity, maxDisparity, mcArray);

        mBWM->disable_debug();

        cout << "Internal buffer size of BilateralWindowMatcher: " 
             << mBWM->get_internal_buffer_szie() / 1024.0 / 1024 << " MB." << endl;

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
}

TEST_F(Test_BilateralWindowMatcher, match_single_line_02)
{
    using namespace std;

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
        mBWM->match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1], 
            halfCount, minDisparity, maxDisparity, mcArray);

        for ( int i = 0; i < pixels; ++i )
        {
            mcArray[i].reset();
        }

        mBWM->enable_debug();
        // mBWM->debug_set_array_buffer_idx( pixels - 1 - halfCount );
        mBWM->debug_set_array_buffer_idx( pixels - 1 );

        mBWM->match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1],
            halfCount, minDisparity, maxDisparity, mcArray);

        mBWM->disable_debug();

        cout << "Internal buffer size of BilateralWindowMatcher: " 
             << mBWM->get_internal_buffer_szie() / 1024.0 / 1024 << " MB." << endl;

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
}

TEST_F(Test_BilateralWindowMatcher, match_single_line_03)
{
    using namespace std;

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
        mBWM->match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1],
            ( kernelSize * numKernels - 1 )/2 + height / 2 - 1, minDisparity, maxDisparity, mcArray);

        for ( int i = 0; i < pixels; ++i )
        {
            mcArray[i].reset();
        }

        // mBWM->enable_debug();
        // mBWM->debug_set_array_buffer_idx(0);

        mBWM->match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1],
            ( kernelSize * numKernels - 1 )/2 + height / 2 - 1, minDisparity, maxDisparity, mcArray);
        cout << "Internal buffer size of BilateralWindowMatcher: " 
             << mBWM->get_internal_buffer_szie() / 1024.0 / 1024 << " MB." << endl;

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
}

TEST_F(Test_BilateralWindowMatcher, match_single_line_04)
{
    using namespace std;

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
        mBWM->match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1], 
            padded[0].rows / 2, minDisparity, maxDisparity, mcArray);

        cout << "Internal buffer size of BilateralWindowMatcher: " 
             << mBWM->get_internal_buffer_szie() / 1024.0 / 1024 << " MB." << endl;

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
}

#endif

TEST_F(Test_BilateralWindowMatcher, match_single_line_05)
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
        
        mBWM->enable_debug();
        mBWM->debug_set_array_buffer_idx(3095, 1);

        mBWM->match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1],
            ( kernelSize * numKernels - 1 )/2 + img0.rows / 2 - 100, minDisparity, maxDisparity, mcArray);
        cout << "Internal buffer size of BilateralWindowMatcher: " 
             << mBWM->get_internal_buffer_szie() / 1024.0 / 1024 << " MB." << endl;

        mBWM->disable_debug();

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
}

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

TEST_F(Test_BilateralWindowMatcher, match_single_line_06)
{
    using namespace std;

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
        
        // mBWM->enable_debug();
        // mBWM->debug_set_array_buffer_idx(19, 0);

        mBWM->match_single_line( padded[0], padded[1], paddedMask[0], paddedMask[1],
            ( windowWidth - 1 )/2 + img0.rows / 2, minDisparity, maxDisparity, mcArray);
        cout << "Internal buffer size of BilateralWindowMatcher: " 
             << mBWM->get_internal_buffer_szie() / 1024.0 / 1024 << " MB." << endl;

        // mBWM->disable_debug();

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
}

}