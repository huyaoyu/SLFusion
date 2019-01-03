#include <exception>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "SLFusion/BilateralWindowMatcher.hpp"

using namespace cv;
using namespace Eigen;
using namespace std;

namespace slf
{

typedef IMatrix_t IM_t;
typedef FMatrix_t FM_t;
typedef Real_t    R_t;

TEST_F( Test_WeightColor, assignment_operator )
{
    // Create the IndexMapper and WeightColor objects.
    IndexMapper im( 3, 13 );
    WeightColor wco0( 13, im.mKnlIdxRow, im.mKnlIdxCol, 0.0 );

    // Create another WeigthColor object.
    WeightColor wco1 = wco0;

    // Save the value in wco0 to temporary variable.
    int temp = wco0.mKnlIdxRow(0, 0);

    // Modify value in wco0.
    wco0.mKnlIdxRow(0, 0) = -1;

    ASSERT_EQ( wco1.mKnlIdxRow(0, 0), temp ) << "Value in wco1 should not change.";
}

TEST_F( Test_WeightColor, average_color_values )
{
    // Read the image file.
    // cout << "Before read image." << endl;
    Mat matTestAvgColorValues = 
        imread("../data/SLFusion/DummyImage_TestAverageColorValues.bmp", IMREAD_COLOR);
    
    if ( matTestAvgColorValues.empty() )
    {
        ASSERT_FALSE(true) << "Read ../data/SLFusion/DummyImage_TestAverageColorValues.bmp failed.";
    }

    // cout << matTestAvgColorValues << endl;
    // cout << "Image read." << endl;

    // Mask.
    Mat mask( matTestAvgColorValues.size(), CV_8UC1 );
    mask.setTo( Scalar::all( 255 ) );

    Mat vcMat;

    Mat matAveragedColorValues;

    // cout << "matTestAvgColorValues.size() = " << matTestAvgColorValues.size() << endl;
    IndexMapper im( 3, 13 );
    WeightColor wco( 13, im.mKnlIdxRow, im.mKnlIdxCol, 0.0 );
    wco.put_average_color_values( matTestAvgColorValues, matAveragedColorValues, mask, vcMat );

    // cout << matAveragedColorValues << endl;

    int lastIdx = 13 - 1;

    ASSERT_EQ(  4, matAveragedColorValues.at<Vec3f>(0, 0)[0] );
    ASSERT_EQ( 13, matAveragedColorValues.at<Vec3f>(0, 0)[1] );
    ASSERT_EQ( 22, matAveragedColorValues.at<Vec3f>(0, 0)[2] );

    ASSERT_EQ(  4, matAveragedColorValues.at<Vec3f>(lastIdx, lastIdx)[0] );
    ASSERT_EQ( 13, matAveragedColorValues.at<Vec3f>(lastIdx, lastIdx)[1] );
    ASSERT_EQ( 22, matAveragedColorValues.at<Vec3f>(lastIdx, lastIdx)[2] );
}

TEST_F( Test_WeightColor, average_color_values_mask )
{
    // Read the image file.
    // cout << "Before read image." << endl;
    Mat matTestAvgColorValues = 
        imread("../data/SLFusion/DummyImage_TestAverageColorValues.bmp", IMREAD_COLOR);

    if ( matTestAvgColorValues.empty() )
    {
        ASSERT_FALSE(true) << "Read ../data/SLFusion/DummyImage_TestAverageColorValues.bmp failed.";
    }

    // cout << "Image read." << endl;

    // The input image looks like this:
    // Channel0   Channel1   Channel2
    //  0, 1, 2   9, 10, 11  18, 19, 20
    //  3, 4, 5  12, 13, 14  21, 22, 23
    //  6, 7, 8  15, 16, 17  24, 25, 26

    // Mask.
    Mat mask( matTestAvgColorValues.size(), CV_8UC1 );
    mask.setTo( Scalar::all( 255 ) );

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

    Mat vcMat, matAveragedColorValues;

    // cout << "matTestAvgColorValues.size() = " << matTestAvgColorValues.size() << endl;
    IndexMapper im( 3, 13 );
    WeightColor wco( 13, im.mKnlIdxRow, im.mKnlIdxCol, 0.0 );
    wco.put_average_color_values( matTestAvgColorValues, matAveragedColorValues, mask, vcMat );

    // cout << matAveragedColorValues << endl;

    int lastIdx = 13 - 1;

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

TEST_F( Test_WeightColor, put_wc_all_the_same )
{
    // Create an input Mat object with all its pixels equal Scalar::all(255).
    const int numKernels  = 13;
    const int windowWidth = 3 * 13;

    Mat src( windowWidth, windowWidth, CV_8UC3 );
    src.setTo( Scalar::all( 255 ) );

    Mat mask( windowWidth, windowWidth, CV_8UC1 );
    mask.setTo( Scalar::all( 1 ) );

    Mat vcMat;

    const int centerIdx = ( numKernels - 1 ) / 2;

    // Resulting matrix.
    FM_t wc = FM_t( numKernels, numKernels );

    // Get the wc values.
    IndexMapper im( 3, 13 );
    WeightColor wco( 13, im.mKnlIdxRow, im.mKnlIdxCol, 13.0 );
    Mat bufferK( numKernels, numKernels, CV_32FC1 );
    wco.wc( src, mask, wc, bufferK );

    // cout << "wc = " << endl << wc << endl;

    ASSERT_EQ( wc( 0, 0 ), 1 );
    ASSERT_EQ( wc( centerIdx, centerIdx ), 1 );
    ASSERT_EQ( wc( numKernels - 1, numKernels - 1 ), 1 );

    ASSERT_EQ( wc.rows(), numKernels ) << "The rows of weight color is numKernels.";
    ASSERT_EQ( wc.cols(), numKernels ) << "The cols of weight color is numKernels.";
}

TEST_F( Test_WeightColor, put_wc_special_center )
{
    // Create an input Mat object with all its pixels equal Scalar::all(255) except the center one.
    const int numKernels  = 13;
    const int windowWidth = 3 * 13;

    Mat src( windowWidth, windowWidth, CV_8UC3 );
    src.setTo( Scalar::all( 255 ) );

    Mat mask( src.size(), CV_8UC1 );
    mask.setTo( Scalar::all( 1 ) );

    const int centerIdxSrc = ( windowWidth - 1 ) / 2;
    const int centerIdx    = ( numKernels - 1 ) / 2;

    ASSERT_EQ( centerIdxSrc, 19 );

    src.at<Vec3b>( centerIdxSrc, centerIdxSrc ) = Vec3b( 1, 1, 1 );

    // Resulting matrix.
    FM_t wc = FM_t( numKernels, numKernels );

    Mat avgColor( numKernels, numKernels, CV_32FC3 );

    // Get the wc values.
    IndexMapper im( 3, 13 );
    WeightColor wco( 13, im.mKnlIdxRow, im.mKnlIdxCol, 13.0 );
    wco.wc( src, mask, wc, avgColor );

    // cout << "avgColor = " << endl << avgColor << endl;

    // cout << "wc = " << endl << wc << endl;

    const R_t nonCenterValue = 0.02327958049488006;
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

    wco.wc( src, mask, wc, avgColor );

    // cout << "wc = " << endl << wc << endl;

    const R_t nonCenterValue2 = 2.0080584471753715e-15;

    ASSERT_EQ( wc( centerIdx,     centerIdx     ), 1 );
    ASSERT_LT( std::fabs( ( wc(centerIdx - 1, centerIdx + 1) - nonCenterValue2 ) / nonCenterValue2 ), eps );
    ASSERT_LT( std::fabs( ( wc(centerIdx + 1, centerIdx - 1) - nonCenterValue2 ) / nonCenterValue2 ), eps );
}

TEST_F(Test_WeightColor, put_wc_mask)
{
    // Create an input Mat object with all its pixels equal Scalar::all(255) except the center one.
    const int numKernels  = 13;
    const int windowWidth = 3 * 13;

    Mat src( windowWidth, windowWidth, CV_8UC3 );
    src.setTo( Scalar::all( 255 ) );

    const int centerIdxSrc = ( windowWidth - 1 ) / 2;
    const int centerIdx    = ( numKernels - 1 ) / 2;

    src.at<Vec3b>( centerIdxSrc, centerIdxSrc ) = Vec3b( 1, 1, 1 );

    src.at<Vec3b>( centerIdxSrc - 1, centerIdxSrc - 1 ) = Vec3b( 1, 1, 1 );
    src.at<Vec3b>( centerIdxSrc - 1, centerIdxSrc - 0 ) = Vec3b( 1, 1, 1 );
    src.at<Vec3b>( centerIdxSrc - 1, centerIdxSrc + 1 ) = Vec3b( 1, 1, 1 );
    src.at<Vec3b>( centerIdxSrc - 0, centerIdxSrc - 1 ) = Vec3b( 1, 1, 1 );
    src.at<Vec3b>( centerIdxSrc - 0, centerIdxSrc + 1 ) = Vec3b( 1, 1, 1 );
    src.at<Vec3b>( centerIdxSrc + 1, centerIdxSrc - 1 ) = Vec3b( 1, 1, 1 );
    src.at<Vec3b>( centerIdxSrc + 1, centerIdxSrc - 0 ) = Vec3b( 1, 1, 1 );
    src.at<Vec3b>( centerIdxSrc + 1, centerIdxSrc + 1 ) = Vec3b( 1, 1, 1 );

    Mat mask( src.size(), CV_8UC1 );
    mask.setTo( Scalar::all( 1 ) );

    mask( Rect( 0, 0, windowWidth, centerIdxSrc ) ).setTo( Scalar::all(0) );
    mask( Rect( 0, centerIdxSrc, centerIdxSrc, centerIdxSrc + 1 ) ).setTo( Scalar::all(0) );

    // Resulting matrix.
    FM_t wc = FM_t( numKernels, numKernels );
    Mat avgColor( numKernels, numKernels, CV_32FC3 );

    // Get the wc values.
    IndexMapper im( 3, 13 );
    WeightColor wco( 13, im.mKnlIdxRow, im.mKnlIdxCol, 13.0 );
    wco.wc( src, mask, wc, avgColor );

    // cout << "avgColor = " << endl << avgColor << endl;
    // cout << "wc = " << endl << wc << endl;

    const R_t centerValue = 1.0;
    const R_t nonCenterValue = 2.0080584471753715e-15;
    const R_t eps = 1e-5;

    ASSERT_LT( std::fabs( ( wc(    centerIdx,     centerIdx) - 1.0 ) / 1.0 ), eps ) << "Center index must always be 1.0.";
    ASSERT_LT( std::fabs( ( wc(    centerIdx, centerIdx + 1) - nonCenterValue ) / nonCenterValue ), eps );
    ASSERT_LT( std::fabs( ( wc(centerIdx + 1, centerIdx + 1) - nonCenterValue ) / nonCenterValue ), eps );
    ASSERT_EQ( wc(centerIdx - 1, centerIdx + 1), 0.0 );
    ASSERT_EQ( wc(centerIdx - 1,     centerIdx), 0.0 );
    ASSERT_EQ( wc(centerIdx - 1, centerIdx - 1), 0.0 );
}

}