/*
 * StereoDRun_StereoDisparity.hpp
 *
 *  Created on: Aug 7, 2018
 *      Author: yyhu
 */

#ifndef INCLUDE_STEREODISPARITY_STEREODISPARITY_HPP_
#define INCLUDE_STEREODISPARITY_STEREODISPARITY_HPP_

#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <boost/exception/all.hpp>
#include <boost/shared_ptr.hpp>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "Runnable/Runnable.hpp"

namespace slf
{

class Run_StereoDisparity : public Runnable
{
public:
    typedef ushort D_t;
	typedef float real;

    typedef struct Vec
    {
        real x;
        real y;
    } Vec_t;

    typedef enum
    {
        SIDE_0 = 0,
        SIDE_1 = 1,
        SIDE_2 = 2,
        SIDE_3 = 3
    } Side_t;

public:
	Run_StereoDisparity();
	~Run_StereoDisparity();

	Runnable::RES_t run(void);

protected:
    Runnable::RES_t put_sides( const Vec_t& r, Side_t& s0, Side_t& s1 );
    void put_starting_points( const Vec_t& r, int H, int W, Vec_t* buffer, int& n );
    void interpolate_along_r(const Vec_t& r, Vec_t& dxdy);
    void draw_along_r(cv::OutputArray _image, const Vec_t& r, const cv::Point& p, int h, int w, const cv::Scalar& color, bool reverse = false);

    Runnable::RES_t read_images( const std::string& fn0, const std::string& fn1 );

    void put_r(std::vector<Vec_t>& vecR, int nAngles);

    void disparity_by_mutual_information(
        cv::InputArray _img0, cv::InputArray _img1, cv::InputArray _mi,
        int dMin, int dMax, cv::OutputArray _D, int nAngles = 16);

public:
    const int IDX_H; // Height index.
    const int IDX_W; // Width index.
    const real SMALL_VALUE;

private:
    cv::Mat mSrcImgs[2];
    cv::Mat mGreyImgs[2];
    cv::Mat mMI;
    cv::Mat mD;           // Disparity map.
};

}

#endif /* INCLUDE_STEREODISPARITY_STEREODISPARITY_HPP_ */