#include "TopCommon.hpp"
#include "SLFusion/CUDA/CUDAMatchImage.cuh"

#include <stdio.h>

#include <iostream>

// #define __CUDA_DEBUG__

namespace slf_cuda
{

dim3 BLOCK_DIM(64, 1, 1);
dim3 GRID_DIM(1, 64, 1); 

typedef struct
{
    int H;
    int W;
    int channels;
    const Intensity_t* mat;
    const Integral_t* imgInt;
    const Integral_t* maskInt;
} Image_t;

}

using namespace slf_cuda;

extern __shared__ CRReal_t crShared[];

/**
 * \param upX The upper left corner x index of the window. X index must be increase in terms of channels.
 * \param upY The upper left corner y index of the window. 
 */
__device__ void put_reference_window( const Image_t* ref, CRReal_t* win, int upX, int upY, int winWidth )
{
    int x = 0;
    const int stride = blockDim.x;
    const int winSize = winWidth * winWidth;

    int y = 0;

    const int imgStride = ref->W * ref->channels;

    int fillStart = 0;

#ifdef __CUDA_DEBUG__
    printf("upX: %d, upY: %d, imgStride: %d\n", upX, upY, imgStride);
#endif
    for ( int i = threadIdx.x; i < winSize; i += stride )
    {
        x = i % winWidth;
        y = i / winWidth;

        if ( y < winWidth )
        {
#ifdef __CUDA_DEBUG__
            printf("x: %d, y: %d\n", x, y);
#endif
            x = (upX + x) * ref->channels;
            y = (upY + y) * imgStride;
            fillStart = i * ref->channels;

#ifdef __CUDA_DEBUG__
            printf("x: %d, y: %d, fillStart: %d\n", x, y, fillStart);
#endif

            for ( int k = 0; k < ref->channels; k++ )
            {
                win[ fillStart + k ] = ref->mat[ y + x + k ];
            }
        }
        else
        {
            break;
        }
    }
}

/**
 * \param H The height of refMat, including the padding.
 * \param W The width of refMat, including the padding.
 * \param channels Is either 1 or 3.
 */
__global__ void __cuda_match_image( 
    int H, int W, int channels,
    const Intensity_t* refMat, const Intensity_t* tstMat, 
    const Integral_t* refInt,  const Integral_t* tstInt, 
    const Integral_t* refMInt, const Integral_t* tstMInt, 
    int kernelSize, int numKernels,
    const CRReal_t* wss,
    int minDisp, int maxDisp, 
    CRReal_t* disp )
{
    // CUDA kernel thread id.
    const int crRow = blockIdx.y * blockDim.y + threadIdx.y;
    // const int crCol = blockIdx.x * blockDim.x + threadIdx.x;
    const int crRowStride = gridDim.y * blockDim.y;
#ifdef __CUDA_DEBUG__
//    printf("gridIdx.x: %d.", gridIdx.x);
#endif
    // Shared memory inside a block.
    const int winWidth = kernelSize * numKernels;
    const int winSize  = winWidth * winWidth;

    CRReal_t* winRef = (CRReal_t*)crShared;
    CRReal_t* wcRef  = (CRReal_t*)( crShared + channels * winSize * sizeof( CRReal_t ) );
    CRReal_t* costs  = (CRReal_t*)( wcRef + winSize * sizeof(CRReal_t) );

    // Local memory for each thread.
    Image_t ref;
    ref.H = H; ref.W = W; ref.channels = channels;
    ref.mat = refMat; ref.imgInt = refInt; ref.maskInt = refMInt;

    Image_t tst;
    tst.H = H; tst.W = W; tst.channels = channels;
    tst.mat = tstMat; tst.imgInt = tstInt; tst.maskInt = tstMInt;

    // Determine the indices of the reference image.
    const int halfWin = ( winWidth - 1 ) / 2;
#ifdef __CUDA_DEBUG__
    printf("crRow: %d, crRowStride: %d\n", crRow, crRowStride);
#endif
    // Loop for every row of the image.
    for ( int row = halfWin + crRow; row < H - halfWin; row += crRowStride )
    {
        // The row this block is processing.
        for ( int col = halfWin + minDisp; col < W - halfWin; col += 1 )
        {
#ifdef __CUDA_DEBUG__
            printf("row: %d, col: %d\n", row, col);
#endif
            // The column in the reference image this block is processing.
            // Every block starts processing from col = halfWin + minDisp.

            // All threads get the pixels from the reference image.
            put_reference_window( &ref, winRef, col, row, winWidth );

            // Make sure that all threads are ready.
            __syncthreads();

            // 
        }
    }
}

int CUDAMatcher::cuda_match_image( 
    int H, int W, int channels,
    const Intensity_t* refMat, const Intensity_t* tstMat, 
    const Integral_t* refInt,  const Integral_t* tstInt, 
    const Integral_t* refMInt, const Integral_t* tstMInt, 
    int kernelSize, int numKernels,
    const CRReal_t* wss,
    int minDisp, int maxDisp, 
    CRReal_t* disp)
{
    // Size of data.
    const int size2D = H * W;             // Size fo the 2D positions.
    const int sizeCh = size2D * channels; // Size considering the channels.
    const int winWidth = kernelSize * numKernels;
    const int sizeWin  = winWidth * winWidth;

    printf("size2D: %d, sizeCh: %d, winWidth: %d, sizeWin: %d\n", size2D, sizeCh, winWidth, sizeWin);

    // Declare all the pointers for memory sharing between CPU and GPU.
    // The "d" prefix is for "device".
    Intensity_t* dRefMat  = NULL;
    Intensity_t* dTstMat  = NULL;
    Integral_t*  dRefInt  = NULL;
    Integral_t*  dTstInt  = NULL;
    Integral_t*  dRefMInt = NULL;
    Integral_t*  dTstMInt = NULL;
    CRReal_t* dWss  = NULL;
    CRReal_t* dDisp = NULL;

    // Allocate managed GPU memory.
    int localMemSize = 0;
    cudaMallocManaged( &dRefMat, sizeCh * sizeof( Intensity_t ) ); localMemSize += sizeCh * sizeof( Intensity_t );
    cudaMallocManaged( &dTstMat, sizeCh * sizeof( Intensity_t ) ); localMemSize += sizeCh * sizeof( Intensity_t );
    cudaMallocManaged( &dRefInt, sizeCh * sizeof( Integral_t ) ); localMemSize += sizeCh * sizeof( Integral_t );
    cudaMallocManaged( &dTstInt, sizeCh * sizeof( Integral_t ) ); localMemSize += sizeCh * sizeof( Integral_t );
    cudaMallocManaged( &dRefMInt, size2D * sizeof( Integral_t ) ); localMemSize += size2D * sizeof( Integral_t );
    cudaMallocManaged( &dTstMInt, size2D * sizeof( Integral_t ) ); localMemSize += size2D * sizeof( Integral_t );
    cudaMallocManaged( &dWss, sizeWin * sizeof( CRReal_t ) ); localMemSize += sizeWin * sizeof( CRReal_t );
    cudaMallocManaged( &dDisp, size2D * sizeof( CRReal_t ) ); localMemSize += size2D * sizeof( CRReal_t );

    printf("Allocated %f MB of managed memory.\n", localMemSize / 1024.0 / 1024);

    for ( int i = 0; i < sizeCh; ++i )
    {
        dRefMat[i] = refMat[i];
        dTstMat[i] = tstMat[i];
        dRefInt[i] = refInt[i];
        dTstInt[i] = tstInt[i];
    }

    for ( int i = 0; i < size2D; ++i )
    {
        dRefMInt[i] = refMInt[i];
        dTstMInt[i] = tstMInt[i];
        dDisp[i] = -1.0;
    }

    for ( int i = 0; i < sizeWin; ++i )
    {
        dWss[i] = wss[i];
    }

    cudaError_t cudaErrorCode = cudaGetLastError();
    if ( cudaSuccess != cudaErrorCode )
    {
        printf("Before call __cuda_match_image, cudaErrorCode %d: %s.\n",
            cudaErrorCode, cudaGetErrorString(cudaErrorCode) );
        return (int)cudaErrorCode; 
    }

    // __cuda_match_image<<<GRID_DIM, BLOCK_DIM>>>( H, W, channels,
    //     dRefMat, dTstMat, dRefInt, dTstInt, dRefMInt, dTstMInt, kernelSize, numKernels, dWss,
    //     minDisp, maxDisp, dDisp );

    // Calculate the external shared memory needed.
    const int crExSharedSize = 
        channels * sizeWin * sizeof( CRReal_t ) + 
        sizeWin * sizeof( CRReal_t ) + 
        ( maxDisp - minDisp + 1 ) * sizeof( CRReal_t );
    
    printf("Shared memory for each block: %d Bytes.\n", crExSharedSize);

    __cuda_match_image<<<dim3(1,3040,1), dim3(64,1,1), (size_t)crExSharedSize>>>( H, W, channels,
        dRefMat, dTstMat, dRefInt, dTstInt, dRefMInt, dTstMInt, kernelSize, numKernels, dWss,
        minDisp, maxDisp, dDisp );

    // Wait for the GPU.
    cudaDeviceSynchronize();

    // Release the managed GPU memory.
    cudaFree( dDisp );       dDisp = NULL;
    cudaFree( dWss );         dWss = NULL;
    cudaFree( dTstMInt ); dTstMInt = NULL;
    cudaFree( dRefMInt ); dRefMInt = NULL;
    cudaFree( dTstInt );   dTstInt = NULL;
    cudaFree( dRefInt );   dRefInt = NULL;
    cudaFree( dTstMat );   dTstMat = NULL;
    cudaFree( dRefMat );   dRefMat = NULL;

    cudaErrorCode = cudaGetLastError();
    if ( cudaSuccess != cudaErrorCode )
    {
        printf(" At last. cudaErrorCode %d: %s.\n",
            cudaErrorCode, cudaGetErrorString(cudaErrorCode) );
        return (int)cudaErrorCode;
    }

    return 0;
}


