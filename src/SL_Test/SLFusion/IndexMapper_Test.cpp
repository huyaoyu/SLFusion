#include <exception>
#include <iostream>
#include <string>

#include "SLFusion/BilateralWindowMatcher.hpp"

using namespace cv;
using namespace Eigen;
using namespace std;

namespace slf
{

typedef IMatrix_t IM_t;

TEST_F(Test_IndexMapper, index_map)
{
    // Create an IndexMapper object.
    IndexMapper im (3, 13);

    // Index maps.
    ASSERT_EQ( im.mIndexMapRow( 0,  0),  1 ) << "mIndexMapRow( 0,  0) == 1";
    ASSERT_EQ( im.mIndexMapRow( 0, 38),  1 ) << "mIndexMapRow( 0, 38) == 1";
    ASSERT_EQ( im.mIndexMapRow(38, 38), 37 ) << "mIndexMapRow(38, 38) == 37";
    ASSERT_EQ( im.mIndexMapRow(38,  0), 37 ) << "mIndexMapRow(38,  0) == 37";
    ASSERT_EQ( im.mIndexMapRow(19, 19), 19 ) << "mIndexMapRow(19, 19) == 19";

    ASSERT_EQ( im.mIndexMapCol( 0,  0),  1 ) << "mIndexMapCol( 0,  0) == 1";
    ASSERT_EQ( im.mIndexMapCol( 0, 38), 37 ) << "mIndexMapCol( 0, 38) == 37";
    ASSERT_EQ( im.mIndexMapCol(38, 38), 37 ) << "mIndexMapCol(38, 38) == 37";
    ASSERT_EQ( im.mIndexMapCol(38,  0),  1 ) << "mIndexMapCol(38,  0) == 1";
    ASSERT_EQ( im.mIndexMapCol(10, 19), 19 ) << "mIndexMapCol(19, 19) == 19";
}

TEST_F(Test_IndexMapper, kernel_index)
{
    // Create an IndexMapper object.
    IndexMapper im(3, 13);

    // Kernel index.
    ASSERT_EQ( im.mKnlIdxRow( 3,  3),  1 ) << "mKnlIdxRow( 3,  3) == 1";
    ASSERT_EQ( im.mKnlIdxRow( 3, 35),  1 ) << "mKnlIdxRow( 3, 35) == 1";
    ASSERT_EQ( im.mKnlIdxRow(35, 35), 11 ) << "mKnlIdxRow(35, 35) == 11";
    ASSERT_EQ( im.mKnlIdxRow(35,  3), 11 ) << "mKnlIdxRow(35, 35) == 11";
    ASSERT_EQ( im.mKnlIdxRow(19, 19),  6 ) << "mKnlIdxRow(19, 19) == 6";

    ASSERT_EQ( im.mKnlIdxCol( 3,  3),  1 ) << "mKnlIdxCol( 3,  3) == 1";
    ASSERT_EQ( im.mKnlIdxCol( 3, 35), 11 ) << "mKnlIdxCol( 3, 35) == 11";
    ASSERT_EQ( im.mKnlIdxCol(35, 35), 11 ) << "mKnlIdxCol(35, 35) == 11";
    ASSERT_EQ( im.mKnlIdxCol(35,  3),  1 ) << "mKnlIdxCol(35,  3) == 1";
    ASSERT_EQ( im.mKnlIdxCol(19, 19),  6 ) << "mKnlIdxCol(19, 19) == 6";
}
TEST_F(Test_IndexMapper, point_index_kernel)
{
    // Create an IndexMapper object.
    IndexMapper im(3, 13);

    // Point index kernel.
    ASSERT_EQ( im.mPntIdxKnlRow( 0,  1),  1 ) << "mPntIdxKnlRow( 0,  1) == 1";
    ASSERT_EQ( im.mPntIdxKnlRow( 1, 12),  4 ) << "mPntIdxKnlRow( 1, 12) == 3";
    ASSERT_EQ( im.mPntIdxKnlRow(12, 11), 37 ) << "mPntIdxKnlRow(12, 11) == 37";
    ASSERT_EQ( im.mPntIdxKnlRow(11,  0), 34 ) << "mPntIdxKnlRow(11,  0) == 34";
    ASSERT_EQ( im.mPntIdxKnlRow( 6,  6), 19 ) << "mPntIdxKnlRow( 6,  6) == 19";

    ASSERT_EQ( im.mPntIdxKnlCol( 0,  1),  4 ) << "mPntIdxKnlCol( 0,  1) == 4";
    ASSERT_EQ( im.mPntIdxKnlCol( 1, 12), 37 ) << "mPntIdxKnlCol( 1, 12) == 37";
    ASSERT_EQ( im.mPntIdxKnlCol(12, 11), 34 ) << "mPntIdxKnlCol(12, 11) == 34";
    ASSERT_EQ( im.mPntIdxKnlCol(11,  0),  1 ) << "mPntIdxKnlCol(11,  0) ==  1";
    ASSERT_EQ( im.mPntIdxKnlCol( 6,  6), 19 ) << "mPntIdxKnlCol( 6,  6) == 19";
}

}