#ifndef __SLFUSION_SLCOMMON_HPP__
#define __SLFUSION_SLCOMMON_HPP__

#include <opencv2/core.hpp>
#include <iostream>

namespace slf
{

std::string ocv_type_literal(int type);
void describe_ocv_mat(const cv::Mat& m, const std::string& name);

/**
 * This function converts the input mat object into CV_8UC1 or CV_8UC3.
 * The input mat object will be duplicated and not modified.
 * 
 * The written file is in JPEG format.
 * 
 * @param fn The filename without the extension.
 */
void write_floating_point_mat_as_byte(const std::string& fn, const cv::Mat& m);

}

#endif // __SLFUSION_SLCOMMON_HPP__