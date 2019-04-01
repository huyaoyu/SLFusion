#ifndef __CUDAROUTINES_COMMON_HPP__
#define __CUDAROUTINES_COMMON_HPP__

#include "TopCommon.hpp"

namespace slf_cuda
{

const int N_SMM = 22;
const int N_SP_PER_SMM = 128;

typedef float CRReal_t;
typedef unsigned char Intensity_t;
typedef int Integral_t;

} /* namespace slf_cuda */

#endif /* __CUDAROUTINES_COMMON_HPP__ */