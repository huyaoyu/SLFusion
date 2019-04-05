#include "TopCommon.hpp"
#include "SLFusion/CUDA/CUDATests.cuh"

#include <stdio.h>

extern __shared__ slf_cuda::CRReal_t crShared[];

namespace slf_cuda
{

__device__ void __cuda_put_window( int H, int W, int channels, 
    const Intensity_t* img, CRReal_t* win, int X, int Y, int winWidth, CRReal_t* out)
{
    int x = 0;
    const int stride = blockDim.x;
    const int winSize = winWidth * winWidth;
    const int half = ( winWidth - 1 ) / 2;

    int y = 0;

    const int imgStride = W * channels;

    int fillStart = 0;
    CRReal_t temp = 0.0;
    CRReal_t sTemp[3] = {0.0, 0.0, 0.0};

    for ( int i = threadIdx.x; i < winSize; i += stride )
    {
        x = i % winWidth;
        y = i / winWidth;

        if ( y < winWidth )
        {
            x = (X - half + x) * channels;
            y = (Y - half + y) * imgStride;
            fillStart = i * channels;

            for ( int k = 0; k < channels; k++ )
            {
                temp = img[ y + x + k ];
                win[ fillStart + k ] = temp;
                sTemp[k] += temp;
            }
        }
        else
        {
            break;
        }
    }

    for( int k = 0; k < channels; ++k )
    {
        atomicAdd( out + Y * imgStride + X * channels + k, sTemp[k] );
    }
}

__global__ void __cuda_test_read_window_from_ocv( int H, int W, int channels,
    const Intensity_t* img, int kernelSize, int numKernels, CRReal_t* out )
{
    // CUDA kernel thread id.
    const int crRow = blockIdx.y * blockDim.y + threadIdx.y;
    // const int crCol = blockIdx.x * blockDim.x + threadIdx.x;
    const int crRowStride = gridDim.y * blockDim.y;

    // Shared memory inside a block.
    const int winWidth = kernelSize * numKernels;
    const int winSize  = winWidth * winWidth;

    CRReal_t* win = (CRReal_t*)crShared;

    // Determine the indices of the image.
    const int halfWin = ( winWidth - 1 ) / 2;

    // Loop for every row of the image.
    for ( int row = halfWin + crRow; row < H - halfWin; row += crRowStride )
    {
        // The row this block is processing.
        for ( int col = halfWin; col < W - halfWin; col += 1 )
        {
            // The column in the image this block is processing.
            // Every block starts processing from col = halfWin.

            // All threads get the pixels from the reference image.
            __cuda_put_window( H, W, channels, 
                img, win, col, row, winWidth, out );

            // Make sure that all threads are ready.
            __syncthreads();
        }
    }
}

int test_read_window_from_ocv(int H, int W, int channels, const Intensity_t* img, 
    int kernelSize, int numKernels, CRReal_t* out)
{
    int res = 0;

    // Size of data.
    const int size2D = H * W;             // Size fo the 2D positions.
    const int sizeCh = size2D * channels; // Size considering the channels.
    const int winWidth = kernelSize * numKernels;
    const int sizeWin  = winWidth * winWidth;

    printf("size2D: %d, sizeCh: %d, winWidth: %d, sizeWin: %d\n", size2D, sizeCh, winWidth, sizeWin);

    // Allocate managed memory.
    Intensity_t* dImg = NULL;
    CRReal_t*    dOut = NULL;

    cudaMallocManaged( &dImg, sizeCh * sizeof(Intensity_t) );
    cudaMallocManaged( &dOut, sizeCh * sizeof(CRReal_t) );

    // Copy data from CPU to GPU.
    for ( int i = 0; i < sizeCh; ++i )
    {
        dImg[i] = img[i];
        dOut[i] = 0.0;
    }

    // Error check.
    cudaError_t cudaErrorCode = cudaGetLastError();
    if ( cudaSuccess != cudaErrorCode )
    {
        printf("Before call __cuda_match_image, cudaErrorCode %d: %s.\n",
            cudaErrorCode, cudaGetErrorString(cudaErrorCode) );
        return (int)cudaErrorCode; 
    }

    // Calculate the block shared memory.
    size_t crExSharedSize = channels * sizeWin * sizeof( CRReal_t );

    // Launch CUDA kernel.
    __cuda_test_read_window_from_ocv<<<dim3(1,3040,1), dim3(128,1,1), crExSharedSize>>>(
        H, W, channels, dImg, 
        kernelSize, numKernels, dOut);

    // Wait for the GPU.
    cudaDeviceSynchronize();

    // Copy values from GPU to CPU.
    for ( int i = 0; i < sizeCh; ++i )
    {
        out[i] = dOut[i];
    }

    cudaFree( dOut ); dOut = NULL;
    cudaFree( dImg ); dImg = NULL;

    cudaErrorCode = cudaGetLastError();
    if ( cudaSuccess != cudaErrorCode )
    {
        printf(" At last. cudaErrorCode %d: %s.\n",
            cudaErrorCode, cudaGetErrorString(cudaErrorCode) );
        return (int)cudaErrorCode;
    }

    return res;
}
    
}

