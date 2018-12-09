/*
 * main.cpp
 *
 *  Created on: Jun 24, 2018
 *      Author: yyhu
 */

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include <gtest/gtest.h>

// #include "InterpolationForStereo/InterpolationForStereo.hpp"
// #include "MutualInformation/MutualInformation.hpp"
// #include "StereoDisparity/StereoDisparity.hpp"

#include "SLFusion/SLFusion.hpp"

int main(int argc, char* argv[])
{
	std::cout << "Test SLFusion." << std::endl;

	// Show sizes of types.
	slf::Runnable::show_size_of_types();

    // GTest.
    testing::InitGoogleTest(&argc, argv);

	return RUN_ALL_TESTS();
}

