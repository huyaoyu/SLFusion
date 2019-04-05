#include "TopCommon.hpp"
#include "SLFusion/CUDA/CUDATests.cuh"

#include <gtest/gtest.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <string>

using namespace cv;
using namespace slf_cuda;

TEST(CUDATests, cuda_read_window_from_ocv)
{
    if ( sizeof(unsigned char) != sizeof(Intensity_t) )
    {
        ASSERT_TRUE(false) << "The type size of Intensity_t (" << sizeof(Intensity_t) << ") is not " << sizeof(unsigned char) << ".";
    }

    if ( sizeof(float) != sizeof(CRReal_t) )
    {
        ASSERT_TRUE(false) << "The type size of CRReal_t (" << sizeof(CRReal_t) << ") is not " << sizeof(float) << ".";
    }

    // The input filename.
    std::string fn = "../data/SLFusion/Img123.bmp";

    // Read the dummy image.
    Mat dummyImg = imread(fn, IMREAD_UNCHANGED);

    std::cout << "Load " << fn << std::endl;
    std::cout << "Image dimension: (" << dummyImg.rows << ", " 
              << dummyImg.cols << ", "
              << dummyImg.channels() << ")." << std::endl;

    // Prepare the memory for the output of the test routine.
    Mat output;
    
    if ( 1 == dummyImg.channels() )
    {
        output = Mat::zeros( dummyImg.rows, dummyImg.cols, CV_32FC1 );
    }
    else
    {
        output = Mat::zeros( dummyImg.rows, dummyImg.cols, CV_32FC3 );
    }
    
    // Call the CUDA test routine.
    int res = 0;
    res = test_read_window_from_ocv(
        dummyImg.rows, dummyImg.cols, dummyImg.channels(),
        dummyImg.ptr<Intensity_t>(), 
        3, 13, 
        output.ptr<CRReal_t>()
    );

    ASSERT_EQ(res, 0) << "test_read_window_from_ocv() failed with res = " << res << ".";

    CRReal_t* s = NULL;
    const int rows = dummyImg.rows;
    const int cols = dummyImg.cols;
    const int channels = dummyImg.channels();
    const int rowStride = cols * channels;
    const int half = (3*13 - 1) / 2;

    s = output.ptr<CRReal_t>();

    for ( int i = 0; i < 2; ++i )
    {
        std::cout << "Sum at r" << half << ", c" << half + i << " = ";

        for ( int k = 0; k < channels; ++k )
        {
            std::cout << *(s + half * rowStride + (half+i)*channels + k) << ", ";
        }

        std::cout << std::endl;
    }

    ASSERT_EQ( s[ half * rowStride + half * channels + 0 ], 1521 );
    ASSERT_EQ( s[ half * rowStride + half * channels + 1 ], 3042 );
    ASSERT_EQ( s[ half * rowStride + half * channels + 2 ], 4563 );

    for ( int i = 0; i < 2; ++i )
    {
        std::cout << "Sum at r" << rows - half - 1 << ", c" << cols - half - 1 + i << " = ";

        for ( int k = 0; k < channels; ++k )
        {
            std::cout << s[ (rows - half - 1) * rowStride + (cols - half - 1) * channels + k ] << ", ";
        }

        std::cout << std::endl;
    }

    ASSERT_EQ( s[ (rows - half - 1) * rowStride + (cols - half - 1) * channels + 0 ], 30420 );
    ASSERT_EQ( s[ (rows - half - 1) * rowStride + (cols - half - 1) * channels + 1 ], 31941 );
    ASSERT_EQ( s[ (rows - half - 1) * rowStride + (cols - half - 1) * channels + 2 ], 33462 );
}

// TEST(CUDATests, cuda_read_window_from_ocv_cpu)
// {
//     if ( sizeof(unsigned char) != sizeof(Intensity_t) )
//     {
//         ASSERT_TRUE(false) << "The type size of Intensity_t (" << sizeof(Intensity_t) << ") is not " << sizeof(unsigned char) << ".";
//     }

//     if ( sizeof(float) != sizeof(CRReal_t) )
//     {
//         ASSERT_TRUE(false) << "The type size of CRReal_t (" << sizeof(CRReal_t) << ") is not " << sizeof(float) << ".";
//     }

//     // The input filename.
//     std::string fn = "../data/SLFusion/Img123.bmp";

//     // Read the dummy image.
//     Mat dummyImg = imread(fn, IMREAD_UNCHANGED);

//     std::cout << "Load " << fn << std::endl;
//     std::cout << "Image dimension: (" << dummyImg.rows << ", " 
//               << dummyImg.cols << ", "
//               << dummyImg.channels() << ")." << std::endl;

//     // Prepare the memory for the output of the test routine.
//     Mat output;
    
//     if ( 1 == dummyImg.channels() )
//     {
//         output = Mat::zeros( dummyImg.rows, dummyImg.cols, CV_32FC1 );
//     }
//     else
//     {
//         output = Mat::zeros( dummyImg.rows, dummyImg.cols, CV_32FC3 );
//     }

//     const int rows = dummyImg.rows;
//     const int cols = dummyImg.cols;
//     const int channels = dummyImg.channels();
//     const int rowStride = cols * channels;
//     const int half = (3*13 - 1) / 2;

//     Intensity_t* m = dummyImg.ptr<Intensity_t>();
//     CRReal_t* s = output.ptr<CRReal_t>();

//     int indexS = 0;
//     int indexM = 0;

//     for ( int row = half; row < rows - half; ++row )
//     {
//         for ( int col = half; col < cols - half; ++col )
//         {
//             indexS = row*rowStride + col*channels;

//             for ( int y = row - half; y <= row + half; ++y )
//             {
//                 for ( int x = col - half; x <= col + half; ++x )
//                 {
//                     indexM = y*rowStride + x*channels;

//                     for ( int k = 0; k < channels; ++k )
//                     {
//                         s[ indexS + k ] += m[ indexM + k ];
//                     }
//                 }
//             }
//         }
//     }

//     for ( int i = 0; i < 2; ++i )
//     {
//         std::cout << "Sum at r" << half << ", c" << half + i << " = ";

//         for ( int k = 0; k < channels; ++k )
//         {
//             std::cout << *(s + half * rowStride + (half+i)*channels + k) << ", ";
//         }

//         std::cout << std::endl;
//     }

//     ASSERT_EQ( s[half * rowStride + half*channels + 0], 1521 );
//     ASSERT_EQ( s[half * rowStride + half*channels + 1], 3042 );
//     ASSERT_EQ( s[half * rowStride + half*channels + 2], 4563 );
// }

