
#ifndef __SLFUSION_BILATERALWINDOWMATCHER_HPP__
#define __SLFUSION_BILATERALWINDOWMATCHER_HPP__

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

class CVType
{
public:
    CVType() {}
    ~CVType() {}

    static int get_real_number_type( int imageType )
    {
        switch ( imageType )
        {
            case CV_8UC1:
            {
                return sizeof(float) == sizeof(Real_t) ? CV_32FC1 : CV_64FC1;
            }
            case CV_8UC3:
            {
                return sizeof(float) == sizeof(Real_t) ? CV_32FC3 : CV_64FC3;
            }
            default:
            {
                EXCEPTION_BAD_ARGUMENT( imageType, "Only supports CV_8UC1 and CV_8UC3." );
            }
        }
    }
};

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

public:
    friend class Test_IndexMapper;

    FRIEND_TEST(Test_IndexMapper, index_map);
    FRIEND_TEST(Test_IndexMapper, kernel_index);
    FRIEND_TEST(Test_IndexMapper, point_index_kernel);
};

class Test_IndexMapper : public ::testing::Test
{
protected:
    static void SetUpTestCase() { }

    static void TearDownTestCase() { }

    virtual void SetUp() { }

    virtual void TearDown() { }
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

    FRIEND_TEST(Test_WeightColor, assignment_operator);
    FRIEND_TEST(Test_WeightColor, mat_divide_3_channels);
    FRIEND_TEST(Test_WeightColor, mat_divide_1_channel);
    FRIEND_TEST(Test_WeightColor, average_color_values);
    FRIEND_TEST(Test_WeightColor, average_color_values_mask);
    FRIEND_TEST(Test_WeightColor, put_wc_all_the_same);
    FRIEND_TEST(Test_WeightColor, put_wc_special_center);
    FRIEND_TEST(Test_WeightColor, put_wc_mask);

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
     * Calculate block/kernel average based on the input
     * integral images.
     * 
     * Type _IT is the data type of the integral images.
     * 
     * refMask and tstMask MUST use 1 to indicate an valid
     * pixel because the inferred refMInt and tstMInt will 
     * be used as the counts for valid pixels inside individual
     * blocks/kernels.
     * 
     * @param refInt Reference image in its integral form.
     * @param tstInt Test image in its integral form.
     * @param refMInt Reference mask in its integral form. OpenCV data type should be CV_32SC1 (int).
     * @param tstMInt Test mask in its integral form. OpenCV data type should be CV_32SC1 (int).
     */
    template <typename _IT> 
    void match_single_line(
        const Mat& refMat, const Mat& tstMat, 
        const Mat& refInt, const Mat& tstInt,
        const Mat& refMInt, const Mat& tstMInt,
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
    void debug_push_index_avg_color(int idx);

private:
    void update_ws(void);

    int num_inner_pixels(int cols, int minDisp, int halfCount);

    /**
     * 
     * _ST is the data type of sint.
     * _DT is the data type of dst.
     * _VT is the data type of vc.
     * 
     * This function uses mint to calculate the quantity of valid pixels
     * inside a block/kernel.
     * 
     * dst MUST have the size of numKernels x numKernels. Its data type
     * MUST be the same with _DT. Its number of channels MUST be the same
     * with sint.
     * 
     * @param sint Input inegral image.
     * @param mint Input integral mask. The OpenCV data type should be CV_32SC1.
     * @param dst Output averaged blocks, its size is numKernels x numKernels.
     * @param vc Valid count of pixels in each block/kernel. MUST be a single channel Mat object.
     * @param row Center row index.
     * @param col Center column index.
     */
    template<typename _ST, typename _DT, typename _VT>
    void block_average_based_on_integral_image(
        const Mat& sint, const Mat& mint, Mat& dst, Mat& vc, 
        int row, int col) const;

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

    /**
     * The user must make sure thant src and dst have compatible dimensions.
     * _T is the type of the elements of src and dst.
     */
    template <typename _T> 
    void expand_block_2_window_mat(const Mat& src, Mat& dst);

    /**
     * The user must make sure thant src and dst have compatible dimensions.
     * _PT: primitive type.
     * _MT: matrix type.
     */
    template <typename _PT, typename _MT> 
    void expand_block_2_window_matrix(const _MT& src, _MT& dst);

    int debug_get_next_index_avg_color(void);
    int debug_get_size_index_avg_color(void);
    void debug_pop_index_avg_color(void);

    void debug_in_loop_wc_avg_color( int i,
        const Mat& ACArrayRef, const Mat& ACArrayTst,
        const Mat& refMat, const Mat& tstMat, const Mat& refMask, const Mat& tstMask,
        const FMatrix_t& WCRef, const FMatrix_t& WCTst);

    void debug_in_loop_cost(int i, 
        Real_t cost, int disp, int idxAvgColorTst, 
        const Mat& winRef, const Mat& winTst, 
        const Mat& ACRef, const Mat& ACTst,
        const FMatrix_t& WCRef, const FMatrix_t& WCTst,
        const FMatrix_t& tad);

private:
    int mKernelSize;
    int mNumKernels; // Number of kernels along one side.
    int mWindowWidth;

    IndexMapper mIM; // Index mapper.

    FMatrix_t mDistanceMap; // A 2D map. Each element of this map records its distance from the center of the window.
    FMatrix_t mWsMap;
    FMatrix_t mPntDistKnl; // A small 2D matrix. Each element of this matrix is the referenced distance from a kernel center to the window center.
    FMatrix_t mWss; // Square of spacial weights.

    Real_t mGammaS;
    Real_t mGammaC;

    Real_t mTAD_T;

    Mat*       mACArrayRef; // AC means average color.
    Mat*       mACArrayTst;
    FMatrix_t* mWCArrayRef; // WC means color weights.
    FMatrix_t* mWCArrayTst;
    int*       mPixelIdxRef;
    int*       mPixelIdxTst;
    size_t     mABSize;     // AB means array buffer.
    size_t     mABMemorySize; // The memory of all array buffers in bytes.

    WeightColor mWCO; // WeightColor object.

    bool mFlagDebug;
    size_t mDebug_ABIdx0;
    size_t mDebug_ABIdx1;
    std::string mDebug_OutDir;

    std::vector<int> mDebug_ACIdx;

public:
    friend class Test_BilateralWindowMatcher;

    FRIEND_TEST(Test_BilateralWindowMatcher, getter_setter);
    FRIEND_TEST(Test_BilateralWindowMatcher, distance_map);
    FRIEND_TEST(Test_BilateralWindowMatcher, ws); // Weight space.
    FRIEND_TEST(Test_BilateralWindowMatcher, wss);
    FRIEND_TEST(Test_BilateralWindowMatcher, half_count);
    FRIEND_TEST(Test_BilateralWindowMatcher, inner_pixels);
    FRIEND_TEST(Test_BilateralWindowMatcher, create_array_buffer);
    FRIEND_TEST(Test_BilateralWindowMatcher, expand_block_2_window_mat);
    FRIEND_TEST(Test_BilateralWindowMatcher, expand_block_2_window_matrix);
    FRIEND_TEST(Test_BilateralWindowMatcher, TADm_same_ref_tst);
    FRIEND_TEST(Test_BilateralWindowMatcher, TADm_same_ref_tst_random);
    FRIEND_TEST(Test_BilateralWindowMatcher, TADm_manual);

#ifndef OMIT_TESTS
    FRIEND_TEST(Test_BilateralWindowMatcher, match_single_line_01);
    FRIEND_TEST(Test_BilateralWindowMatcher, match_single_line_02);
    FRIEND_TEST(Test_BilateralWindowMatcher, match_single_line_03);
    FRIEND_TEST(Test_BilateralWindowMatcher, match_single_line_04);
#endif
    FRIEND_TEST(Test_BilateralWindowMatcher, match_single_line_05);
    FRIEND_TEST(Test_BilateralWindowMatcher, match_single_line_06);
    FRIEND_TEST(Test_BilateralWindowMatcher, match_single_line_gradient);
    FRIEND_TEST(Test_BilateralWindowMatcher, match_single_line_mb_tsukuba);

};

template<typename _TR, typename _TT> 
void BilateralWindowMatcher::TADm(const Mat& ref, const Mat& tst, FMatrix_t& tad)
{
    // The rows and cols of ref and tst are assumed to be the same.
    const int channels = ref.channels();
    const _TR* pRef    = NULL;
    const _TT* pTst    = NULL;

    Real_t temp = 0.0;

    int posCol = 0, posColShift = 0;
    int posTad = 0;
    // Clear tad.
    tad.setConstant(0.0);

    for ( int i = 0; i < ref.rows; ++i )
    {
        // Get the pointer to the ref and tst.
        pRef = ref.ptr<_TR>(i);
        pTst = tst.ptr<_TT>(i);

        posCol = 0;

        for ( int j = 0; j < ref.cols; ++j )
        {
            temp        = 0.0;
            posColShift = posCol;

            for ( int k = 0; k < channels; ++k)
            {
                temp += 
                ( pRef[posColShift] - pTst[posColShift] ) * 
                ( pRef[posColShift] - pTst[posColShift] );

                posColShift++;
            }

            *(tad.data() + posTad) = std::min( std::sqrt(temp), mTAD_T );

            posTad++;
            posCol += channels;
        }
    }
}

template <typename _T> 
void BilateralWindowMatcher::expand_block_2_window_mat(const Mat& src, Mat& dst)
{
    if ( dst.rows != src.rows * mKernelSize ||
         dst.cols != src.cols * mKernelSize )
    {
        std::stringstream ssSrc, ssDst;
        ssSrc << "src.size() = " << src.size();
        ssDst << "dst.size() = " << dst.size();
        EXCEPTION_DIMENSION_MISMATCH(src, ssSrc.str(), dst, ssDst.str());
    }

    const int channels = src.channels();

    const _T* pS = NULL;
    _T*       pD = NULL;
    int posCol   = 0;
    int* pKIR    = mIM.mKnlIdxRow.data(); // Pointer to kernel index row.
    int* pKIC    = mIM.mKnlIdxCol.data(); // Pointer to kernel index column.
    int iKIR     = 0;
    int iKIC     = 0;
    int posKI    = 0;

    for ( int i = 0; i < dst.rows; ++i )
    {
        pD = dst.ptr<_T>(i);

        posCol = 0;
        for ( int j = 0; j < dst.cols; ++j )
        {
            iKIR = *( pKIR + posKI );
            iKIC = *( pKIC + posKI );
            pS   = src.ptr<_T>(iKIR);

            for ( int k = 0; k < channels; ++k )
            {
                *( pD + posCol + k) = pS[iKIC * channels + k];
            }

            posKI  += 1;
            posCol += channels;
        }
    }
}

template <typename _PT, typename _MT> 
void BilateralWindowMatcher::expand_block_2_window_matrix(const _MT& src, _MT& dst)
{
    if ( dst.size() != mIM.mKnlIdxRow.size() )
    {
        // This is an error.
        std::stringstream ssDst, ssMIM;
        ssDst << "dst.size() = " << dst.size();
        ssMIM << "mIM.mKnlIdxRow.size() = " << mIM.mKnlIdxRow.size();
        EXCEPTION_DIMENSION_MISMATCH(dst, ssDst.str(), mIM, ssMIM.str() );
    }

    const _PT* pS = src.data();
    _PT*       pD = dst.data();
    const int  cS = src.cols(); // Colmumn number of src.

    int* pKIR = mIM.mKnlIdxRow.data();
    int* pKIC = mIM.mKnlIdxCol.data();
    int  iKIR = 0;
    int  iKIC = 0;

    for ( int i = 0; i < dst.size(); ++i )
    {
        iKIR = *( pKIR + i );
        iKIC = *( pKIC + i );

        *( pD + i ) = pS[ iKIR * cS + iKIC ];
    }
}

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

    static void create_gradient_image( OutputArray _dst, int height, int width, 
        const std::vector<int>& b, const std::vector<int>& g, const std::vector<int>& r );

protected:
    static const int mDefaultKernelSize;
    static const int mDefaultNumKernels;
    static const int mDefaultWindowWidth;
};

} // namespace slf.

#endif // __SLFUSION_BILATERALWINDOWMATCHER_HPP__
