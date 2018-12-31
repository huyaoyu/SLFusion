#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "SLFusion/SLCommon.hpp"

namespace slf
{

std::string ocv_type_literal(int type)
{
    int quotient  = type / 8;
    int remainder = type % 8;

    std::string strChannel;

    switch ( quotient )
    {
        case 0:
        {
            strChannel = "C1";
            break;
        }
        case 1:
        {
            strChannel = "C2";
            break;
        }
        case 2:
        {
            strChannel = "C3";
            break;
        }
        case 3:
        {
            strChannel = "C4";
            break;
        }
        default:
        {
            std::cout << "Error: Unexpected type number " << type << std::endl;
        }
    }

    std::string strType;

    switch ( remainder )
    {
        case 0:
        {
            strType = "CV_8U";
            break;
        }
        case 1:
        {
            strType = "CV_8S";
            break;
        }
        case 2:
        {
            strType = "CV_16U";
            break;
        }
        case 3:
        {
            strType = "CV_16S";
            break;
        }
        case 4:
        {
            strType = "CV_32S";
            break;
        }
        case 5:
        {
            strType = "CV_32F";
            break;
        }
        case 6:
        {
            strType = "CV_64F";
            break;
        }
        default:
        {
            std::cout << "Error: Unexpected type number " << type << std::endl;
        }
    }

    std::string temp = strType + strChannel;

    return temp;
}

void describe_ocv_mat(const cv::Mat& m, const std::string& name)
{
    std::cout << name << ".type()     = " << m.type() << " (" << ocv_type_literal(m.type()) << ")" <<  std::endl;
    std::cout << name << ".size()     = " << m.size()     << std::endl;
    std::cout << name << ".rows       = " << m.rows       << std::endl;
    std::cout << name << ".cols       = " << m.cols       << std::endl;
    std::cout << name << ".channels() = " << m.channels() << std::endl;
}

void write_floating_point_mat_as_byte(const std::string& fn, const cv::Mat& m)
{
    const int channels = m.channels();

    cv::Mat dst;

    // Convert the type according to the channels.
    if ( 3 == channels )
    {
        m.convertTo( dst, CV_8UC3 );
    }
    else if ( 1 == channels )
    {
        m.convertTo( dst, CV_8UC1 );
    }
    else
    {
        // Error.
        std::cout << "Error: write_floating_point_mat_as_byte: Unexpected number of channels: " << channels << std::endl;
        return;
    }

    std::vector<int> jpegParam{ cv::IMWRITE_JPEG_QUALITY, 100 };

    cv::imwrite( fn + ".jpg", dst, jpegParam );
}

}
