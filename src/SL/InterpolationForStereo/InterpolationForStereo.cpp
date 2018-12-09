
#define _USE_MATH_DEFINES

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <sstream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "InterpolationForStereo/InterpolationForStereo.hpp"

using namespace slf;

static Run_InterpolationForStereo::Vec_t
get_r_by_angle(Run_InterpolationForStereo::real angle)
{
    // Local struct variable.
    Run_InterpolationForStereo::Vec_t r;
    r.x = cos(angle);
    r.y = sin(angle);

    return r;
}

Run_InterpolationForStereo::Run_InterpolationForStereo()
: Runnable("InterpolationForStereo"),
  IDX_H(0), IDX_W(1), SMALL_VALUE(1e-6),
  mNR(16)
{

}

Run_InterpolationForStereo::~Run_InterpolationForStereo()
{

}

void Run_InterpolationForStereo::set_image_size(int h, int w)
{
    mImageSize[IDX_H] = h;
    mImageSize[IDX_W] = w;
}

void Run_InterpolationForStereo::set_n_angles(int n)
{
    mNR = n;
}

Runnable::RES_t Run_InterpolationForStereo::put_sides( const Vec_t& r, Side_t& s0, Side_t& s1 )
{
    real x = ( fabs(r.x) < SMALL_VALUE ) ? 0.0 : r.x;
    real y = ( fabs(r.y) < SMALL_VALUE ) ? 0.0 : r.y;

    if ( 0.0 == x && 0.0 == y )
    {
        std::cout << "The norm of r is zero." << std::endl;
        return Runnable::ERROR;
    }

    if ( x < 0.0 )
    {
        if ( y < 0.0 )
        {
            s0 = SIDE_1;
            s1 = SIDE_2;
        }
        else if ( 0.0 == y )
        {
            s0 = SIDE_1;
            s1 = SIDE_1;
        }
        else
        {
            s0 = SIDE_0;
            s1 = SIDE_1;
        }
    }
    else if ( 0.0 == x )
    {
        if ( y < 0.0 )
        {
            s0 = SIDE_2;
            s1 = SIDE_2;
        }
        else if ( 0.0 == y )
        {
            // Never reached.    
        }
        else
        {
            s0 = SIDE_0;
            s1 = SIDE_0;
        }
    }
    else
    {
        if ( y < 0.0 )
        {
            s0 = SIDE_2;
            s1 = SIDE_3;
        }
        else if ( 0.0 == y )
        {
            s0 = SIDE_3;
            s1 = SIDE_3;
        }
        else
        {
            s0 = SIDE_3;
            s1 = SIDE_0;
        }
    }

    return Runnable::OK;
}

void Run_InterpolationForStereo::interpolate_along_r(const Vec_t& r, Vec_t& dxdy)
{
    real frx = fabs(r.x);
    real fry = fabs(r.y);

    if ( frx > fry )
    {
        dxdy.x = r.x / frx;
        dxdy.y = r.y / frx;
    }
    else if ( frx < fry )
    {
        dxdy.x = r.x / fry;
        dxdy.y = r.y / fry;
    }
    else
    {
        dxdy.x = r.x / frx;
        dxdy.y = r.y / fry;
    }
}

void Run_InterpolationForStereo::draw_along_r(cv::OutputArray _image, const Vec_t& r, const cv::Point& p, int h, int w, const cv::Scalar& color, bool reverse)
{
    // Get dx and dy.

    Vec_t coor;
    coor.x = p.x;
    coor.y = p.y;

    Vec_t dxdy;
    interpolate_along_r(r, dxdy);

    cv::Mat image = _image.getMat();

    cv::Point pos;

    while ( 1 )
    {
        // Shift.
        if ( false == reverse )
        {
            coor.x += dxdy.x;
            coor.y += dxdy.y;
        }
        else
        {
            coor.x -= dxdy.x;
            coor.y -= dxdy.y;
        }

        pos.x = (int)(coor.x);
        pos.y = (int)(coor.y);

        if ( pos.x < 0 || pos.x > w - 1 ||
             pos.y < 0 || pos.y > h - 1 )
        {
            break;
        }

        // Draw one pixel.
        image.at<cv::Vec3b>(pos) = cv::Vec3b(color[0], color[1], color[2]);

        // std::cout << "pos(" << pos.x << ", " << pos.y << ")" << std::endl;
    }
}

Runnable::RES_t Run_InterpolationForStereo::run(void)
{
	Runnable::RES_t res = Runnable::OK;

	this->show_header();

	// Output dir.
    std::string outDir = "../data/InterpolationForStereo";

    // Populate mR;

    real* angleArray = new real[mNR];
    Vec_t r;

    real angleStep = 2 * M_PI / mNR;

    for ( int i = 0; i < mNR; ++i )
    {
        r = get_r_by_angle( 1.0 * i * angleStep );
        mR.push_back(r);
    }

    Side_t s0, s1;

    std::cout << "The content of mR is: " << std::endl;

    std::vector<Vec_t>::iterator iter;
    std::cout << "r.x, r.y, s0, s1" << std::endl;
    std::cout << "[ " << std::endl;
    std::cout << std::scientific;
    std::cout << std::showpos;
    for ( iter = mR.begin(); iter != mR.end(); ++iter )
    {
        put_sides((*iter), s0, s1);
        std::cout << (*iter).x << ", " << (*iter).y 
                  << ", " << s0 << ", " << s1 << std::endl;
    }
    std::cout << std::noshowpos;
    std::cout.unsetf(std::ios::scientific);
    std::cout << "]" << std::endl;

    // Line color.
    cv::Scalar red(0, 0, 255);
    cv::Scalar black(0);

    // Starting points.
    std::vector<cv::Point> startingPoints;
    startingPoints.push_back( cv::Point(               0,               0 ));
    startingPoints.push_back( cv::Point( mImageSize[1]-1,               0 ));
    startingPoints.push_back( cv::Point( mImageSize[1]-1, mImageSize[0]-1 ));
    startingPoints.push_back( cv::Point(               0, mImageSize[0]-1 ));
    startingPoints.push_back( cv::Point( mImageSize[1]/2,               0 ));
    startingPoints.push_back( cv::Point( mImageSize[1]-1, mImageSize[0]/2 ));
    startingPoints.push_back( cv::Point( mImageSize[1]/2, mImageSize[0]-1 ));
    startingPoints.push_back( cv::Point(               0, mImageSize[0]/2 ));
    startingPoints.push_back( cv::Point( mImageSize[1]/2, mImageSize[0]/2 ));
    startingPoints.push_back( cv::Point( mImageSize[1]/3, mImageSize[0]/3 ));

    // Parameters for saving.
    std::vector<int> jpgParams;
    jpgParams.push_back(cv::IMWRITE_JPEG_QUALITY);
    jpgParams.push_back(100);

    // Temporary image.
    cv::Mat image;

    // Looping indices.
    int idx      = 0;
    int idxImage = 0;

    // Looping string stream.
    std::stringstream ss;

    // Loop over the staring points.
    std::vector<cv::Point>::iterator iterStartingPoint;

    for ( iterStartingPoint = startingPoints.begin(); 
          iterStartingPoint != startingPoints.end(); 
          ++iterStartingPoint )
    {
        // Clear string stream.
        ss.clear();
        ss.str("");

        std::cout << "Image #" << idxImage << std::endl;

        // Create an image with solid color as the background.
        image = cv::Mat(mImageSize[IDX_H], mImageSize[IDX_W], CV_8UC3, black);

        // Interpolate on the image.
        idx = 0;
        for ( iter = mR.begin(); iter != mR.end(); ++iter )
        {
            std::cout << "Draw line " << idx << std::endl;
            draw_along_r(image, *iter, *iterStartingPoint, mImageSize[0], mImageSize[1], red);
            idx++;
        }

        // Save the image.
        ss << outDir << "/path_" << idxImage << ".jpg";
        cv::imwrite(ss.str().c_str(), image, jpgParams);

        // Show the iamge.
        cv::namedWindow("Image", cv::WINDOW_NORMAL);
        cv::imshow("Image", image);
        cv::waitKey(500);

        std::cout << std::endl;

        idxImage++;
    }

    delete [] angleArray; angleArray = NULL;

	this->show_footer();

	return res;
}
