
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

#include "SLFusion/SLFusion.hpp"

using namespace slf;

const float SMALL_VALUE_LOG = 1e-30;

static void show_floating_point_number_image(cv::InputArray _src, const std::string& winName, const char* outPathFileName = NULL)
{
	cv::Mat src = _src.getMat();

	namedWindow(winName.c_str(), cv::WINDOW_NORMAL);
	double minV = 0.0, maxV = 0.0;
	cv::minMaxLoc(src, &minV, &maxV);

	cv::Mat shifted = (src - minV) / (maxV - minV);

	cv::minMaxLoc(shifted, &minV, &maxV);

	cv::imshow(winName.c_str(), shifted);

	if ( NULL != outPathFileName )
	{
		cv::Mat converted = shifted * 255;
//		converted.convertTo(converted, CV_8UC1);
		cv::imwrite(outPathFileName, converted);
	}
}

template<typename T>
void flush_small_positive_values(cv::InputOutputArray _m, T limit, T v = -1)
{
	if ( v <= 0 )
	{
		v = limit;
	}

	cv::Mat m = _m.getMat();

	for ( int i = 0; i < m.rows; ++i )
	{
		for ( int j = 0; j< m.cols; ++j )
		{
			if ( m.at<T>(i, j) <= limit )
			{
				m.at<T>(i, j) = v;
			}
		}
	}
}

static void put_initial_disparity_map(int rows, int cols, cv::OutputArray _D, float val, int type = CV_32FC1)
{
	if ( type == CV_32FC1 || type == CV_32F ||
		 type == CV_64FC1 || type == CV_64F )
	{
		_D.create( rows, cols, type );

		cv::Mat D = _D.getMat();

		D = val;
	}
	else
	{
		std::cout << "Error. Type must be one of the followings: CV_32FC1, CV_32F, CV_64FC1, CV_64F." << std::endl;
		EXCEPTION_BAD_OCV_DATA_TYPE(type, "CV_32FC1, CV_32F, CV_64FC1, CV_64F", type);
	}
}

/***
 * \param _src Must be using the type of CV_8UC1.
 */
template<typename T>
static void warp(cv::InputArray _src, cv::InputArray _D, cv::OutputArray _dst)
{
	// Check if _src is of type CV_8UC1.
	cv::Mat src = _src.getMat();
	int srcType = src.type();

	if ( CV_8UC1 != srcType )
	{
		std::cout << "Error. Input mat must be of type CV_8UC1." << std::endl;
		EXCEPTION_BAD_OCV_DATA_TYPE(_src, "CV_8UC1", srcType);
	}

	// Check the dimensions.
	cv::Mat D = _D.getMat();

	if ( src.size != D.size )
	{
		std::cout << "Error. The source image and the D map should have the same sizes." << std::endl;
		std::stringstream ssSizeSrc, ssSizeD;
		ssSizeSrc << "(" << src.size[0] << ", " << src.size[1] << ")";
		ssSizeD   << "(" << D.size[0] << ", " << D.size[1] << ")";
		EXCEPTION_DIMENSION_MISMATCH(src, ssSizeSrc.str(), D, ssSizeD.str());
	}

	// Allocate memory for _dst.
	_dst.create( src.rows, src.cols, CV_8UC1 );
	cv::Mat dst = _dst.getMat();

	src.copyTo(dst);

	int d  = 0; // Disparity.
	int j2 = 0; // Newly projected index.

	for ( int i = 0; i < src.rows; ++i )
	{
		for ( int j = 0; j < src.cols; ++j )
		{
			// Disparity.
			d = (int)( D.at<T>(i,j) );

			if ( d <= 0 )
			{
				d = 0;
			}
			else if ( j + d >= src.cols  )
			{
				d = src.cols - j;
			}

			// New index.
			j2 = j + d;

			// Assign
			dst.at<uchar>(i, j2) = src.at<uchar>(i, j);
		}
	}
}

static Run_SLFusion::Vec_t
get_r_by_angle(Run_SLFusion::real angle)
{
    // Local struct variable.
    Run_SLFusion::Vec_t r;
    r.x = cos(angle);
    r.y = sin(angle);

    return r;
}

static void
put_index_matching_window(int h, int w, int xr, int xt, int winWidth)
{
    
}

Run_SLFusion::Run_SLFusion()
: Runnable("SLFusion"),
  IDX_H(0), IDX_W(1), SMALL_VALUE(1e-6)
{

}

Run_SLFusion::~Run_SLFusion()
{

}

Runnable::RES_t Run_SLFusion::put_sides( const Vec_t& r, Side_t& s0, Side_t& s1 )
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

void Run_SLFusion::put_starting_points( const Vec_t& r, int H, int W, Vec_t* buffer, int& n )
{
    // Get the two possible sides.
	Side_t s0, s1;
	put_sides(r, s0, s1);

	int idxBuffer = 0;
	n = 0;

	switch(s0)
	{
		case SIDE_0:
		{
			for( int i = 0; i < W; ++i ) { buffer[idxBuffer].x = i; buffer[idxBuffer].y = 0; idxBuffer++; }
			break;
		}
		case SIDE_1:
		{
			for( int i = 0; i < H; ++i ) { buffer[idxBuffer].x = 0; buffer[idxBuffer].y = i; idxBuffer++; }
			break;
		}
		case SIDE_2:
		{
			for( int i = W - 1; i > 0; --i ) { buffer[idxBuffer].x = i; buffer[idxBuffer].y = 0; idxBuffer++; }
			break;
		}
		case SIDE_3:
		{
			for( int i = H - 1; i > 0; --i ) { buffer[idxBuffer].x = 0; buffer[idxBuffer].y = i; idxBuffer++; }
			break;
		}
		default:
		{
			// Should never reach here.
		}
	}

	n += idxBuffer;

	if ( s1 == s0 )
	{
		// That's it. 
		return;
	}

	idxBuffer = n;

	switch(s1)
	{
		case SIDE_0:
		{
			for( int i = 0; i < W; ++i ) { buffer[idxBuffer].x = i; buffer[idxBuffer].y = 0; idxBuffer++; }
			break;
		}
		case SIDE_1:
		{
			for( int i = 0; i < H; ++i ) { buffer[idxBuffer].x = 0; buffer[idxBuffer].y = i; idxBuffer++; }
			break;
		}
		case SIDE_2:
		{
			for( int i = W - 1; i > 0; --i ) { buffer[idxBuffer].x = i; buffer[idxBuffer].y = 0; idxBuffer++; }
			break;
		}
		case SIDE_3:
		{
			for( int i = H - 1; i > 0; --i ) { buffer[idxBuffer].x = 0; buffer[idxBuffer].y = i; idxBuffer++; }
			break;
		}
		default:
		{
			// Should never reach here.
		}
	}
}

void Run_SLFusion::interpolate_along_r(const Vec_t& r, Vec_t& dxdy)
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

void Run_SLFusion::draw_along_r(cv::OutputArray _image, const Vec_t& r, const cv::Point& p, int h, int w, const cv::Scalar& color, bool reverse)
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

Runnable::RES_t Run_SLFusion::read_images( const std::string& fn0, const std::string& fn1 )
{
	// Read these two images.
	mSrcImgs[0] = imread( fn0, cv::IMREAD_COLOR );
	mSrcImgs[1] = imread( fn1, cv::IMREAD_COLOR );

	// Check the image size of these images.
    if ( mSrcImgs[0].size != mSrcImgs[1].size )
    {
        std::cout << "The image sizes are not compatible." << std::endl;
        this->show_footer();
        return Runnable::ERROR;
    }

    std::cout << "The image size is (" 
              << mSrcImgs[0].size[0] << ", "
              << mSrcImgs[0].size[1] << ")" << std::endl;

	// Generate grey scale images.
	cv::cvtColor(mSrcImgs[0], mGreyImgs[0], cv::COLOR_BGR2GRAY, 1);
	cv::cvtColor(mSrcImgs[1], mGreyImgs[1], cv::COLOR_BGR2GRAY, 1);

	return Runnable::OK;
}

Runnable::RES_t Run_SLFusion::read_LIDAR( const std::string& fn )
{
    // Use cnpy.
    try 
    {
        mArrLIDARMap = cnpy::npy_load(fn);
    }
    catch ( std::runtime_error& e )
    {
        std::cout << "cnpy::npy_load fails with " << e.what() << "." << std::endl;
        return Runnable::ERROR;
    }
    
    // Convert the data into cv::Mat.
    double* loadedData = mArrLIDARMap.data<double>();
    mLIDARMap = cv::Mat( 
        mArrLIDARMap.shape[0],
        mArrLIDARMap.shape[1], CV_64FC1, loadedData );

    int positiveCount = 0;

    for ( int i=0; i < mLIDARMap.rows; ++i )
    {
        for ( int j=0; j < mLIDARMap.cols; ++j )
        {
            if ( mLIDARMap.at<double>(i, j) > 0.0 )
            {
                positiveCount++;
            }
        }
    }

    // Report the sum of all elements for data checking.
    double temp = cv::sum(mLIDARMap)[0];
    std::cout << "The sum of all elements of " << fn << " is " << temp << " with " << positiveCount << " positive values." << std::endl;

    return Runnable::OK;
}

void Run_SLFusion::put_r(std::vector<Vec_t>& vecR, int nAngles)
{
	real* angleArray = new real[nAngles];
    Vec_t r;

    real angleStep = 2 * M_PI / nAngles;

    for ( int i = 0; i < nAngles; ++i )
    {
        r = get_r_by_angle( 1.0 * i * angleStep );
        vecR.push_back(r);
    }

	delete [] angleArray; angleArray = NULL;
}

Runnable::RES_t Run_SLFusion::run(void)
{
	Runnable::RES_t res = Runnable::OK;

	this->show_header();

	// Filenames.
	std::string filenames[3];
	filenames[0] = "../data/SLFusion/L.jpg";
	filenames[1] = "../data/SLFusion/R.jpg";
	filenames[2] = "../data/SLFusion/dmL.npy";

	try
	{
		if ( Runnable::OK != read_images( filenames[0], filenames[1] ) )
		{
			std::cout << "Could not read the files." << std::endl;
			std::string expString = filenames[0];
			expString += " and ";
			expString += filenames[1];

			EXCEPTION_FILE_OPEN_FAILED(expString);
		}

        if ( Runnable::OK != read_LIDAR( filenames[2] ) )
		{
			std::cout << "Could not read the files." << std::endl;
			std::string expString = filenames[2];

			EXCEPTION_FILE_OPEN_FAILED(expString);
		}

		// Show the grey images.
		cv::namedWindow("Greyscale image left", cv::WINDOW_NORMAL );
		cv::imshow("Greyscale image left", mGreyImgs[0] );
		cv::namedWindow("Greyscale image right", cv::WINDOW_NORMAL );
		cv::imshow("Greyscale image right", mGreyImgs[1] );

		// Warp operation.
		cv::Mat warped; // Warped image.

		put_initial_disparity_map( mGreyImgs[0].rows, mGreyImgs[0].cols, mD, 600.0, CV_32FC1 );

		warp<Run_SLFusion::real>( mGreyImgs[1], mD, warped );

		cv::namedWindow("Warped image", cv::WINDOW_NORMAL);
		cv::imshow("Warped image", warped);
	}
	catch ( exception_base& exp )
	{
		std::cout << diagnostic_information(exp);
	}
	
	cv::waitKey(0);

	MemSize::show_memory_usage();

	this->show_footer();

	return res;
}
