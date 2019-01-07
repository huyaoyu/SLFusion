#include <cmath>
#include <fstream>
#include <vector>

#include <opencv2/highgui.hpp>

#include "SLFusion/BilateralWindowMatcher.hpp"

using namespace cv;
using namespace Eigen;

namespace slf
{

template <typename _AT> 
void WeightColor::wc( const Mat& ac, const Mat& vc, FMatrix_t& wc )
{
    // NOTE: wc has to be row-major to maintain the performance.
    const _AT* pAc     = NULL;
    int pos            = 0;
    int posCol         = 0;
    const int channels = ac.channels();
    Real_t colorDiff   = 0.0;
    Real_t colorDist   = 0.0; // Color distance. L2 distance.
    Real_t colorAc[3]  = { 0.0, 0.0, 0.0 };
    Real_t* pWC        = wc.data();
    const uchar* pVC   = NULL;

    // Get the color at the center.
    pAc = ac.ptr<_AT>( mCenterIdx );
    for ( int i = 0; i < channels; ++i )
    {
        colorAc[i] = *( pAc + mCenterIdx*channels + i );
    }

    // Calculate the weight color values.
    for ( int i = 0; i < ac.rows; ++i )
    {
        pAc    = ac.ptr<_AT>( i );
        pVC    = vc.ptr<uchar>(i);
        posCol = 0;

        for ( int j = 0; j < ac.cols; ++j )
        {
            if ( 0 != *(pVC + j) )
            {
                colorDist = 0.0;

                for ( int k = 0; k < channels; ++k )
                {
                    colorDiff = 
                        colorAc[k] - *( pAc + posCol + k );
                    
                    colorDist += colorDiff * colorDiff;
                }

                colorDist = std::sqrt( colorDist );

                *( pWC + pos ) = std::exp( -colorDist / mGammaC );
            }
            else
            {
                *( pWC + pos ) = 0.0;
            }

            posCol += channels;
            pos++;
        }
    }
}

}