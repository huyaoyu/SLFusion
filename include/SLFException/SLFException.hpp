/*
 * LOCVException.hpp
 *
 *  Created on: Aug 7, 2018
 *      Author: yyhu
 */

#ifndef INCLUDE_SLFEXCEPTION_SLFEXCEPTION_HPP_
#define INCLUDE_SLFEXCEPTION_SLFEXCEPTION_HPP_

#include <exception>
#include <iostream>
#include <sstream>

#include <boost/exception/all.hpp>
#include <boost/shared_ptr.hpp>

#define EXCEPTION_ARG_OUT_OF_RANGE(v, minV, maxV) \
    {\
        std::stringstream v##_ss;\
        v##_ss << "Argument out of range, " \
               << #v << " = " << v \
               << ", [" << minV << ", " << maxV << "]. "\
               << "Value not changed.";\
        BOOST_THROW_EXCEPTION( argument_out_of_range() << ExceptionInfoString(v##_ss.str()) );\
    }

#define EXCEPTION_FILE_OPEN_FAILED(fn) \
    {\
        std::stringstream exp_ss;\
        exp_ss << "File " << fn << " open failed.";\
        BOOST_THROW_EXCEPTION( file_open_failed() << ExceptionInfoString( exp_ss.str() ) );\
    }

#define EXCEPTION_BAD_OCV_DATA_TYPE(v, expectedType, actualType) \
    {\
        std::stringstream v##_ss;\
        v##_ss << "Expect " << #v \
               << " to be of type " << expectedType \
               << ". But get " << actualType << " instead.";\
        BOOST_THROW_EXCEPTION( bad_ocv_data_type() << ExceptionInfoString( v##_ss.str() ) );\
    }

#define EXCEPTION_DIMENSION_MISMATCH(v1, dimString1, v2, dimString2) \
    {\
        std::stringstream exp_ss;\
        exp_ss << "Dimesion mismatch. " \
               << #v1 << " is " << dimString1 << " and " \
               << #v2 << " is " << dimString2;\
        BOOST_THROW_EXCEPTION( dimension_mismatch() << ExceptionInfoString( exp_ss.str() ) );\
    }

namespace slf
{

struct exception_base        : virtual std::exception, virtual boost::exception { };
struct bad_argument          : virtual exception_base { };
struct argument_out_of_range : virtual bad_argument { };

struct bad_ocv_data_type     : virtual exception_base { };
struct dimension_mismatch    : virtual exception_base { };

struct file_open_failed      : virtual exception_base { };

typedef boost::error_info<struct tag_info_string, std::string> ExceptionInfoString;

}

#endif /* INCLUDE_SLFEXCEPTION_SLFEXCEPTION_HPP_ */