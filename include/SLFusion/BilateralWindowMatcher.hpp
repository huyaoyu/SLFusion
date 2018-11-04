
#ifndef __SLFUSION_BILATERALWINDOWMATCHER_HPP__
#define __SLFUSION_BILATERALWINDOWMATCHER_HPP__

#include <iostream>
#include <sstream>
#include <string>

#include <eigen3/Eigen/Dense>

#include "SLFException/SLFException.hpp"

// Name space delaration.
using Eigen::MatrixXd;
using Eigen::MatrixXi;

namespace slf
{

class BilateralWindowMatcher
{
public:
    BilateralWindowMatcher(int w, int nw);
    ~BilateralWindowMatcher(void);

    void show_index_maps(void);

    int get_kernel_size(void);
    int get_num_kernels_single_side(void);

protected:
    int mKernelSize;
    int mNumKernels; // Number of kernels along one side.

    MatrixXi mIndexMapRow; // A 2D map. Each element of this map records its central row index in the original window.
    MatrixXi mIndexMapCol; // A 2D map. Each element of this map records its central col index in the original window.

    MatrixXi mKnlIdxRow; // A 2D reference map. Each element of this map records its row index in the gridded, small matrix.
    MatrixXi mKnlIdxCol; // A 2D reference map. Each element of this map records its col index in the gridded, small matrix.

    MatrixXi mKnlPntIdxRow; // A 2D index matrix. Each element is the original row index of a single kernel center.
    MatrixXi mKnlPntIdxCol; // A 2D index matrix. Each element is the original col index of a single kernel center.

    MatrixXd mDistanceMap; // A 2D map. Each element of this map records its distance from the center of the window.
    MatrixXd mDistanceRef; // A small 2D matrix. Each element of this matrix is the referenced distance from a kernel center to the window center.
};

} // namespace slf.

#endif // __SLFUSION_BILATERALWINDOWMATCHER_HPP__
