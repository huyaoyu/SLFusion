/*
 * SLFusion.hpp
 *
 *  Created on: Sep 18, 2018
 *      Author: yyhu
 */

#ifndef INCLUDE_SLFUSION_SLFUSION_HPP_
#define INCLUDE_SLFUSION_SLFUSION_HPP_

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <boost/exception/all.hpp>
#include <boost/shared_ptr.hpp>

#include <gtest/gtest.h>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

// The cnpy package is requried.
// Find it at
// https://github.com/rogersce/cnpy
#include "cnpy.h"

#include "Runnable/Runnable.hpp"
#include "BilateralWindowMatcher.hpp"

#define BEGIN_IMAGE_LOOP \
    for ( int _idx = 0; _idx < 2; _idx++ ) \
    {

#define END_IMAGE_LOOP \
    }

namespace slf
{

class Run_SLFusion : public Runnable
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
    Run_SLFusion();
	Run_SLFusion(int w, int nw);
	~Run_SLFusion();

	Runnable::RES_t run(void);

    Runnable::RES_t read_images( const std::string& fn0, const std::string& fn1 );

    /** Read the LIDAR data.
     * 
     * The LIdAR data is prepared in the NumPy binary format (.npy).
     * 
     * \param fn The input .npy file.
     * 
     */
    Runnable::RES_t read_LIDAR( const std::string& fn );

    static int put_padded_mat(cv::InputArray _src, int w, int nw, cv::Scalar& spv, cv::OutputArray _dst, cv::OutputArray _mask, double m = 1);

protected:
    Runnable::RES_t put_sides( const Vec_t& r, Side_t& s0, Side_t& s1 );
    void put_starting_points( const Vec_t& r, int H, int W, Vec_t* buffer, int& n );
    void interpolate_along_r(const Vec_t& r, Vec_t& dxdy);
    void draw_along_r(cv::OutputArray _image, const Vec_t& r, const cv::Point& p, int h, int w, const cv::Scalar& color, bool reverse = false);

    void put_r(std::vector<Vec_t>& vecR, int nAngles);

public:
    const int IDX_H; // Height index.
    const int IDX_W; // Width index.
    const real SMALL_VALUE;

private:
    cv::Mat mSrcImgs[2];
    cv::Mat mGreyImgs[2];
    cv::Mat mCIELabImgs[2];

    BilateralWindowMatcher* mBWM;

    cnpy::NpyArray mArrLIDARMap;
    cv::Mat mLIDARMap;

    cv::Mat mD;           // Disparity map.

public:
    friend class Test_SLFusion;
    FRIEND_TEST(Test_SLFusion, OneLineStereoCost);
};

class Test_SLFusion : public ::testing::Test
{
protected:
    static void SetUpTestCase()
    {
        
    }

    static void TearDownTestCase()
    {
        
    }

    virtual void SetUp()
    {
        
    }

    virtual void TearDown()
    {
        
    }
};

}

#endif /* INCLUDE_SLFUSION_SLFUSION_HPP_ */