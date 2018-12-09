#include <iostream>

#include <opencv2/highgui.hpp>

#include "SLFusion/BilateralWindowMatcher.hpp"

using namespace std;

namespace slf
{

BilateralWindowMatcher* Test_BilateralWindowMatcher::mBWM = NULL;

TEST_F( Test_BilateralWindowMatcher, average_color_values )
{
    // Read the image file.
    cout << "Before read image." << endl;
    cv::Mat matTestAvgColorValues = 
        cv::imread("/home/yaoyu/SourceCodes/SLFusion/data/SLFusion/DummyImage_TestAverageColorValues.bmp", cv::IMREAD_COLOR);
    cout << "Image read." << endl;

    cv::Mat matAveragedColorValues;

    cout << "matTestAvgColorValues.size() = " << matTestAvgColorValues.size() << endl;

    // mBWM = new BilateralWindowMatcher( 3, 13 );

    mBWM->put_average_color_values( matTestAvgColorValues, matAveragedColorValues );

    // delete mBWM; mBWM = NULL;

    cout << matAveragedColorValues << endl;
}


}