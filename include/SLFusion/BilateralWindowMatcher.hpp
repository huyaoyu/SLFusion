
#ifndef __SLFUSION_BILATERALWINDOWMATCHER_HPP__
#define __SLFUSION_BILATERALWINDOWMATCHER_HPP__

#include <iostream>
#include <sstream>
#include <string>

#include <eigen3/Eigen/Dense>
#include <opencv2/core.hpp>

#include <gtest/gtest.h>

#include "SLFException/SLFException.hpp"

// Name space delaration.
using namespace cv;
using namespace Eigen;

namespace slf
{

template<typename _T>
class MatchingCost
{
public:
    MatchingCost()
    : mIdxRef(-1), mNTst(0), mIdxTstArray(NULL), mCostArray(NULL)
    {

    }

    ~MatchingCost()
    {
        destory();
    }

    void destory(void)
    {
        if ( NULL != mCostArray )
        {
            delete [] mCostArray; mCostArray = NULL;
        }

        if ( NULL != mIdxTstArray )
        {
            delete [] mIdxTstArray; mIdxTstArray = NULL;
        }

        mNTst = 0;
    }

    void allocate(int n)
    {
        if ( n != mNTst )
        {
            destory();
            mNTst = n;
        }
        
        mIdxTstArray = new int[mNTst];
        mCostArray   = new _T[mNTst];
    }

    void set_idx_ref(int idx)
    {
        mIdxRef = idx;
    }

    int get_idx_ref(void) const
    {
        return mIdxRef;
    }

    int get_n_test(void) const
    {
        return mNTst;
    }

    int* get_p_idx_test(void) const
    {
        return mIdxTstArray;
    }

    _T* get_p_cost(void) const
    {
        return mCostArray;
    }

protected:
    int  mIdxRef;
    int  mNTst;
    int* mIdxTstArray;
    _T*  mCostArray;
};

class BilateralWindowMatcher
{
public:
    typedef float Real_t;
    const int OCV_F_TYPE; // Must compatible with Real_t.
    typedef Matrix<   int, -1, -1, RowMajor> IMatrix_t;
    typedef Matrix<Real_t, -1, -1, RowMajor> FMatrix_t;

public:
    BilateralWindowMatcher(int w, int nw);
    ~BilateralWindowMatcher(void);

    void show_index_maps(void);

    int get_kernel_size(void);
    int get_num_kernels_single_side(void);
    int get_window_width(void);

    void set_gamma_s(Real_t gs);
    Real_t get_gamma_s(void);

    void set_gamma_c(Real_t gc);
    Real_t get_gamma_c(void);

    /**
     * @param refMat Reference image (left).
     * @param tstMat Test image (right).
     * @param rosIdx Row index in refMat and tstMat to calculate the matching cost.
     * @param minDisp Minimum disparity value.
     * @param maxDisp Maximum disparity value.
     * @param pMC Pointer to a pre-allocated array of MachingCost<Real_t> objects.
     * @param nMC The number of MachingCost<Real_t> objects stored starting from pMC. nMC must be smaller than the capacity of pMC. Pass NULL if not interested.
     */
    void match_single_line(
        const Mat& refMat, const Mat& tstMat, int rowIdx,
        int minDisp, int maxDisp, 
        MatchingCost<Real_t>* pMC, int* nMC = NULL);

protected:
    /**
     * @param _src The dimension is (mKernelSize * mNumKernels)^2. _src is assumed to have data type CV_8UC1 or CV_8UC3.
     * @param _dst The dimension is mNumKernels^2. The data type of _dst is either CV_32FC1 or CV_32FC3.
     */
    void put_average_color_values(InputArray _src, OutputArray _dst);

    /**
     * @param src Data type is CV_8UC1 or CV_8UC3.
     * @param bufferS A Mat buffer which has the same height and width of the support window.
     * 
     */
    void put_wc(const Mat& src, FMatrix_t& wc, Mat& avgColor, Mat* bufferS = NULL);

    template<typename tR, typename tT> Real_t TAD( const tR* pr, const tT* pt, int channels );
    void TADm(const Mat& ref, const Mat& tst, FMatrix_t& tad);

protected:
    int mKernelSize;
    int mNumKernels; // Number of kernels along one side.
    int mWindowWidth;

    IMatrix_t mIndexMapRow; // A 2D map. Each element of this map records its central row index in the original window.
    IMatrix_t mIndexMapCol; // A 2D map. Each element of this map records its central col index in the original window.

    IMatrix_t mKnlIdxRow; // A 2D reference map. Each element of this map records its row index in the gridded, small matrix.
    IMatrix_t mKnlIdxCol; // A 2D reference map. Each element of this map records its col index in the gridded, small matrix.

    IMatrix_t mPntIdxKnlRow; // A 2D index matrix. Each element is the original row index of a single kernel center.
    IMatrix_t mPntIdxKnlCol; // A 2D index matrix. Each element is the original col index of a single kernel center.

    FMatrix_t mDistanceMap; // A 2D map. Each element of this map records its distance from the center of the window.
    FMatrix_t mWsMap;

    FMatrix_t mPntDistKnl; // A small 2D matrix. Each element of this matrix is the referenced distance from a kernel center to the window center.

    Real_t mGammaS;
    Real_t mGammaC;

    Real_t mTAD_T;

public:
    friend class Test_BilateralWindowMatcher;
    FRIEND_TEST(Test_BilateralWindowMatcher, average_color_values);
    FRIEND_TEST(Test_BilateralWindowMatcher, put_wc_01);
    FRIEND_TEST(Test_BilateralWindowMatcher, put_wc_02);
};

class Test_BilateralWindowMatcher : public ::testing::Test
{
protected:
    static void SetUpTestCase()
    {
        mBWM = new BilateralWindowMatcher( 3, 13 );
    }

    static void TearDownTestCase()
    {
        if ( NULL != mBWM )
        {
            delete mBWM; mBWM = NULL;
        }
    }

    virtual void SetUp()
    {
        // if ( NULL != mBWM )
        // {
        //     delete mBWM;
        // }

        // mBWM = new BilateralWindowMatcher( 3, 13 );
    }

    virtual void TearDown()
    {
        // if ( NULL != mBWM )
        // {
        //     delete mBWM; mBWM = NULL;
        // }
    }

protected:
    static BilateralWindowMatcher* mBWM;
};

} // namespace slf.

#endif // __SLFUSION_BILATERALWINDOWMATCHER_HPP__
