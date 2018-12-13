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

    Mat matAveragedColorValues;

    // cout << "matTestAvgColorValues.size() = " << matTestAvgColorValues.size() << endl;

    mBWM->put_average_color_values( matTestAvgColorValues, matAveragedColorValues );

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

    const int centerIdx = ( numKernels - 1 ) / 2;

    // Resulting matrix.
    FM_t wc = FM_t( numKernels, numKernels );

    Mat bufferK( numKernels, numKernels, mBWM->OCV_F_TYPE );

    // Get the wc values.
    mBWM->put_wc( src, wc, bufferK, NULL );

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

    const int centerIdxSrc = ( windowWidth - 1 ) / 2;
    const int centerIdx    = ( numKernels - 1 ) / 2;

    ASSERT_EQ( centerIdxSrc, 19 );

    src.at<Vec3b>( centerIdxSrc, centerIdxSrc ) = Vec3b( 1, 1, 1 );

    // Resulting matrix.
    FM_t wc = FM_t( numKernels, numKernels );

    Mat bufferK( numKernels, numKernels, mBWM->OCV_F_TYPE );

    // Get the wc values.
    mBWM->put_wc( src, wc, bufferK, NULL );

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

    mBWM->put_wc( src, wc, bufferK, &bufferS );

    // cout << "wc = " << endl << wc << endl;

    const R_t nonCenterValue2 = 4.930302754597667e-09;

    ASSERT_EQ( wc( centerIdx,     centerIdx     ), 1 );
    ASSERT_LT( std::fabs( ( wc(centerIdx - 1, centerIdx + 1) - nonCenterValue2 ) / nonCenterValue2 ), eps );
    ASSERT_LT( std::fabs( ( wc(centerIdx + 1, centerIdx - 1) - nonCenterValue2 ) / nonCenterValue2 ), eps );
}

TEST_F(Test_BilateralWindowMatcher, match_single_line_01)
{
    using namespace std;

    // Read 1 image.
    string fn = "../data/SLFusion/L.jpg";
    Mat img;

    try
    {
        img = imread( fn, cv::IMREAD_COLOR );

        cout << fn << " is read." << endl 
             << "img.rows = " << img.rows << ", "
             << "img.cols = " << img.cols 
             << endl;
        
        // Padding.
        Mat padded;
        Scalar s = Scalar(0, 0, 0);
        if ( 0 != Run_SLFusion::put_padded_mat( img, 3, 13, s, padded) )
        {
            ASSERT_FALSE(true);
        }

        // Define the disparity range.
        const int minDisparity = 500;
        const int maxDisparity = 999;
        const int numDisparity = maxDisparity - minDisparity + 1;
        const int pixels       = img.cols - minDisparity - 38;

        // Pre-allocations.
        MatchingCost<R_t>* mcArray = new MatchingCost<R_t>[ pixels ];
        for ( int i = 0; i < pixels; ++i )
        {
            mcArray[i].allocate(numDisparity);
        }

        cout << "Estimated memory: " << mcArray[0].estimate_storage() * (pixels) / 1024.0 / 1024 << " MB." << endl;

        // Use this only image as both the reference and test images.
        // Calculate matching cost.
        mBWM->match_single_line( img, img, 19, minDisparity, maxDisparity, mcArray);

        // Verify the matching cost.
    }
    catch ( exception& exp )
    {
        cout << "wat() " << exp.what() << endl;
        ASSERT_FALSE(true);
    }
}

}