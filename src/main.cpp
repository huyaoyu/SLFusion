/*
 * main.cpp
 *
 *  Created on: Jun 24, 2018
 *      Author: yyhu
 */

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

// #include "InterpolationForStereo/InterpolationForStereo.hpp"
// #include "MutualInformation/MutualInformation.hpp"
// #include "StereoDisparity/StereoDisparity.hpp"

#include "SLFusion/SLFusion.hpp"

#define CREATE_ADD_OBJECT(c, n, v) \
	c n;\
	v.push_back( &n );

int main(void)
{
	std::cout << "Hello OpenCV!" << std::endl;

	// Show sizes of types.
	slf::Runnable::show_size_of_types();

	// Create a vector.
	std::vector<slf::Runnable*> vecRunnables;

	// // Create a MutualInformation object.
	// CREATE_ADD_OBJECT(slf::Run_MutualInformation, runMutualInformation, vecRunnables);

	// // Create a InterpolationForStereo objec.
	// CREATE_ADD_OBJECT(slf::Run_InterpolationForStereo, runInterpolationForStereo, vecRunnables);
	// runInterpolationForStereo.set_image_size(3008, 4112);

	// Create a StereoDisparity objec.
	// CREATE_ADD_OBJECT(slf::Run_StereoDisparity, runStrereoDisparity, vecRunnables);

	// Create a SLFusion object.
	CREATE_ADD_OBJECT(slf::Run_SLFusion, runSLFusion, vecRunnables);

	// Run the Runnable object.
	std::vector<slf::Runnable*>::iterator iter;

	for( iter = vecRunnables.begin(); iter != vecRunnables.end(); ++iter )
	{
		(*iter)->run();
	}

	return 0;
}

