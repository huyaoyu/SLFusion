
#ifndef __SLFUSION_BILATERALWINDOWMATCHER_HPP__
#define __SLFUSION_BILATERALWINDOWMATCHER_HPP__

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <eigen3/Eigen/Dense>
#include <opencv2/core.hpp>

#include <gtest/gtest.h>

#include "SLFException/SLFException.hpp"

#define OMIT_TESTS \
    do { } while(0)

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
    : mIdxRef(-1), mSize(0), mNTst(0), mDispArray(NULL), mCostArray(NULL),
      mDataUnit( sizeof(int) + sizeof(_T) )
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

        if ( NULL != mDispArray )
        {
            delete [] mDispArray; mDispArray = NULL;
        }

        mNTst = 0;
        mSize = 0;
    }

    void allocate(int n)
    {
        if ( n != mSize )
        {
            destory();
            mSize = n;
        }
        
        mDispArray = new int[mSize];
        mCostArray = new _T[mSize];
    }

    void push_back(int disp, _T cost)
    {
        if ( mNTst == mSize )
        {
            std::cout << "MatchingCost object reaches the maximum capacity." << std::endl;
            EXCEPTION_BASE("MatchingCost object reaches the maximum capacity.");
        }

        mDispArray[mNTst] = disp;
        mCostArray[mNTst] = cost;
        mNTst++;
    }

    void reset(void)
    {
        mNTst = 0;
    }

    int estimate_storage(void)
    {
        return mDataUnit * mSize;
    }

    void set_idx_ref(int idx)
    {
        mIdxRef = idx;
    }

    int get_idx_ref(void) const
    {
        return mIdxRef;
    }

    int siez(void) const
    {
        return mSize;
    }

    int get_n_test(void) const
    {
        return mNTst;
    }

    int* get_disparity_array(void) const
    {
        return mDispArray;
    }

    _T* get_p_cost(void) const
    {
        return mCostArray;
    }

    /**
     * @param dn The directory name.
     */
    void write(const std::string& dn)
    {
        // Filename.
        std::stringstream ss;
        ss.str(""); ss.clear();
        ss << dn << "/" << std::setfill('0') << std::setw(5) << mIdxRef << ".dat";

        std::ofstream ofs;
        ofs.open( ss.str() );

        ofs.precision(12);
        for ( int i = 0; i < mNTst; ++i )
        {
            ofs << std::showpos << std::scientific << mDispArray[i] << " " << mCostArray[i] << std::endl;
        }

        ofs.close();
    }

protected:
    int  mIdxRef;
    int  mSize;
    int  mNTst;
    int* mDispArray;
    _T*  mCostArray;

    const int  mDataUnit;
};

typedef float Real_t;
typedef Matrix<   int, -1, -1, RowMajor> IMatrix_t;
typedef Matrix<Real_t, -1, -1, RowMajor> FMatrix_t;

class IndexMapper
{
public:
    IndexMapper(int w, int nw);
    ~IndexMapper();

    static void
    put_index_map(IMatrix_t& ri, IMatrix_t& ci, 
        IMatrix_t& rri, IMatrix_t& rci, 
        IMatrix_t& knlRfnIdxRow, IMatrix_t& knlRfnIdxCol, 
        int w);

    void show(void);

public:
    IMatrix_t mIndexMapRow; // A 2D map. Each element of this map records its central row index in the original window.
    IMatrix_t mIndexMapCol; // A 2D map. Each element of this map records its central col index in the original window.

    IMatrix_t mKnlIdxRow; // A 2D reference map. Each element of this map records its row index in the gridded, small matrix.
    IMatrix_t mKnlIdxCol; // A 2D reference map. Each element of this map records its col index in the gridded, small matrix.

    IMatrix_t mPntIdxKnlRow; // A 2D index matrix. Each element is the original row index of a single kernel center.
    IMatrix_t mPntIdxKnlCol; // A 2D index matrix. Each element is the original col index of a single kernel center.
};

class WeightColor
{
public:
    WeightColor(); // This default constructor is only for implicit initialization.
    WeightColor(int numKernels, IMatrix_t& knlIdxRow, IMatrix_t& knlIdxCol, Real_t gammaC);
    ~WeightColor();

    void wc(const Mat& src, const Mat& mask, FMatrix_t& wc, Mat& avgColor);

    WeightColor& operator=( const WeightColor& rhs );

private:
    /**
     * @param _src The dimension is (mKernelSize * mNumKernels)^2. _src is assumed to have data type CV_8UC1 or CV_8UC3.
     * @param _dst The dimension is mNumKernels^2. The data type of _dst is either CV_32FC1 or CV_32FC3.
     */
    void put_average_color_values(
        InputArray _src, OutputArray _dst, InputArray _mask, OutputArray _validCount);

private:
    int mNumKernels;
    int mCenterIdx;
    Real_t mGammaC;
    IMatrix_t mKnlIdxRow; // A 2D reference map. Each element of this map records its row index in the gridded, small matrix.
    IMatrix_t mKnlIdxCol; // A 2D reference map. Each element of this map records its col index in the gridded, small matrix.
    Mat mVcMat;
    Mat mTvcMat;

public:
    friend class Test_WeightColor;

    FRIEND_TEST(Test_WeightColor, average_color_values);
    FRIEND_TEST(Test_WeightColor, put_wc_01);
    FRIEND_TEST(Test_WeightColor, put_wc_02);

};

class Test_WeightColor : public ::testing::Test
{
protected:
    static void SetUpTestCase() { }

    static void TearDownTestCase() { }

    virtual void SetUp() { }

    virtual void TearDown() { }
};

class BilateralWindowMatcher
{
public:
    const int OCV_F_TYPE; // Must compatible with Real_t.
 
public:
    BilateralWindowMatcher(int w, int nw);
    ~BilateralWindowMatcher(void);

    void show_index_maps(void);

    int get_kernel_size(void) const;
    int get_num_kernels_single_side(void) const;
    int get_window_width(void) const;

    void set_gamma_s(Real_t gs);
    Real_t get_gamma_s(void) const;

    void set_gamma_c(Real_t gc);
    Real_t get_gamma_c(void) const;

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
        const Mat& refMat, const Mat& tstMat, 
        const Mat& refMask, const Mat& tstMask,
        int rowIdx,
        int minDisp, int maxDisp, 
        MatchingCost<Real_t>* pMC, int* nMC = NULL);

    /**
     * @return Buffer size is returned as number of bytes. 
     */
    size_t get_internal_buffer_szie(void) const;

    void enable_debug(void);
    void disable_debug(void);
    void debug_set_array_buffer_idx(size_t idx0, size_t idx1 = 0);
    void debug_set_out_dir(const std::string& dir);

private:
    template<typename tR, typename tT> Real_t TAD( const tR* pr, const tT* pt, int channels );

    template<typename _TR, typename _TT> void TADm(const Mat& ref, const Mat& tst, FMatrix_t& tad);

    /**
     * If size <= mABSize, the arrays will be re-used. If size > mABSize, the
     * array buffers will be deleted first and then allocated. Use argument force to
     * explicitly delete and allocated new memory. Newly allocated memory is NOT
     * initialized. If Argument size == 0, then the arrya buffers will be deleted and no
     * new memory will be allocated.
     */
    void create_array_buffer(size_t size, int matType, bool force = false);
    void destroy_array_buffer(void);
    void allocate_array_buffer(size_t size, int matType);

private:
    int mKernelSize;
    int mNumKernels; // Number of kernels along one side.
    int mWindowWidth;

    IndexMapper mIM; // Index mapper.

    FMatrix_t mDistanceMap; // A 2D map. Each element of this map records its distance from the center of the window.
    FMatrix_t mWsMap;
    FMatrix_t mWss; // Square of spacial weights.

    FMatrix_t mPntDistKnl; // A small 2D matrix. Each element of this matrix is the referenced distance from a kernel center to the window center.

    Real_t mGammaS;
    Real_t mGammaC;

    Real_t mTAD_T;

    Mat*       mACArrayRef; // AC means average color.
    Mat*       mACArrayTst;
    FMatrix_t* mWCArrayRef; // WC means color weights.
    FMatrix_t* mWCArrayTst;
    size_t     mABSize;     // AB means array buffer.
    size_t     mABMemorySize; // The memory of all array buffers in bytes.

    WeightColor mWCO; // WeightColor object.

    bool mFlagDebug;
    size_t mDebug_ABIdx0;
    size_t mDebug_ABIdx1;
    std::string mDebug_OutDir;

public:
    friend class Test_BilateralWindowMatcher;

    FRIEND_TEST(Test_BilateralWindowMatcher, getter_setter);
    FRIEND_TEST(Test_BilateralWindowMatcher, half_count);
    FRIEND_TEST(Test_BilateralWindowMatcher, inner_pixels);
    FRIEND_TEST(Test_BilateralWindowMatcher, create_array_buffer);

#ifndef OMIT_TESTS
    FRIEND_TEST(Test_BilateralWindowMatcher, match_single_line_01);
    FRIEND_TEST(Test_BilateralWindowMatcher, match_single_line_02);
    FRIEND_TEST(Test_BilateralWindowMatcher, match_single_line_03);
    FRIEND_TEST(Test_BilateralWindowMatcher, match_single_line_04);
#endif
    FRIEND_TEST(Test_BilateralWindowMatcher, match_single_line_05);
    FRIEND_TEST(Test_BilateralWindowMatcher, match_single_line_06);
};

class Test_BilateralWindowMatcher : public ::testing::Test
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
    static const int mDefaultKernelSize;
    static const int mDefaultNumKernels;
    static const int mDefaultWindowWidth;
};

} // namespace slf.

#endif // __SLFUSION_BILATERALWINDOWMATCHER_HPP__
