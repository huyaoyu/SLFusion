#include "TopCommon.hpp"

#include <iostream>

#include <eigen3/Eigen/Dense>
#include <gtest/gtest.h>

#include "SLFusion/CUDA/TryKernelEigen.cuh"

using namespace Eigen;
using namespace slf_cuda;

TEST(SLFusion_CUDA, try_kernel_eigen)
{
    const int rows  = 1000;
    const int cols  = 1000;
    const int cStep = 3;

    std::cout << "Allocate memory." << std::endl;

    // Prepare a matrix as 1000 x 1000 single-point floating number.
    Matrix<float, Dynamic, Dynamic, RowMajor> ix = Matrix<float, Dynamic, Dynamic, RowMajor>::Zero( rows, cols*cStep );
    Matrix<float, Dynamic, Dynamic, RowMajor> iy = Matrix<float, Dynamic, Dynamic, RowMajor>::Zero( rows, cols*cStep );

    std::cout << "ix(" << ix.rows() << ", " << ix.cols() << ")" << std::endl;
    std::cout << "iy(" << iy.rows() << ", " << iy.cols() << ")" << std::endl;

    for ( int i = 0; i < rows; i += 1)
    {
        for ( int j = 1; j < cols*cStep; j += 3 )
        {
            // std::cout << "(" << i << ", " << j << ")" << std::endl;
            ix(i, j)   = 1;
            ix(i, j+1) = 2;
        }
    }

    std::cout << "Call crExponent()." << std::endl;

    int res = crExponent( ix.data(), rows, cols, cStep, iy.data() );

    std::cout << "iy(0, 0) = " << iy(0, 0) << "." << std::endl;
    std::cout << "iy(1, 1) = " << iy(1, 1) << "." << std::endl;
    std::cout << "iy(2, 2) = " << iy(2, 2) << "." << std::endl;
    std::cout << "iy(2, 3) = " << iy(2, 3) << "." << std::endl;
    std::cout << "iy(2, 6) = " << iy(2, 6) << "." << std::endl;
    std::cout << "iy(3, 3) = " << iy(3, 3) << "." << std::endl;
    std::cout << "iy(3, 6) = " << iy(3, 6) << "." << std::endl;
    std::cout << "iy(0, 1) = " << iy(0, 1) << "." << std::endl;
    std::cout << "iy(0, 2) = " << iy(0, 2) << "." << std::endl;
    std::cout << "iy(0, 3) = " << iy(0, 3) << "." << std::endl;
    std::cout << "iy(0, 4) = " << iy(0, 4) << "." << std::endl;
    std::cout << "iy(0, 5) = " << iy(0, 5) << "." << std::endl;
    std::cout << "iy(0, 6) = " << iy(0, 6) << "." << std::endl;
    std::cout << "iy(1, 0) = " << iy(1, 0) << "." << std::endl;
    std::cout << "iy(1, 1) = " << iy(1, 1) << "." << std::endl;
    std::cout << "iy(1, 2) = " << iy(1, 2) << "." << std::endl;
    std::cout << "iy(1, 3) = " << iy(1, 3) << "." << std::endl;
    std::cout << "iy(1, 4) = " << iy(1, 4) << "." << std::endl;
    std::cout << "iy(1, 5) = " << iy(1, 5) << "." << std::endl;
    std::cout << "iy(1, 6) = " << iy(1, 6) << "." << std::endl;
    std::cout << "iy(0, 2997) = " << iy(0, 2997) << "." << std::endl;
    std::cout << "iy(999,  999) = " << iy(999,  999) << "." << std::endl;
    std::cout << "iy(999, 1000) = " << iy(999, 1000) << "." << std::endl;
    std::cout << "iy(999, 1001) = " << iy(999, 1001) << "." << std::endl;
    std::cout << "iy(38, 114) = " << iy(38, 114) << "." << std::endl;

    ASSERT_EQ( 1521, iy(38, 114) ) << "Window size is 39.";
}