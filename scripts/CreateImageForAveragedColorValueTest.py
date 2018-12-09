#!python

from __future__ import print_function

import cv2
import numpy as np

KERNEL_SIZE = 3
NUM_KERNELS = 13

if __name__ == "__main__":
    print( "Create a dummy image for testing the average color value function." )

    img = np.zeros( (KERNEL_SIZE * NUM_KERNELS, KERNEL_SIZE * NUM_KERNELS, 3), dtype = np.uint8 )

    block = np.zeros( ( KERNEL_SIZE, KERNEL_SIZE, 3 ), dtype = np.uint8 )

    count = 0

    for k in range(3):
        for i in range(KERNEL_SIZE):
            for j in range(KERNEL_SIZE):
                block[i, j, k] = count
                count += 1

    print("The block is ")
    print(block)

    for i in range(NUM_KERNELS):
        rowIdxStart = i * KERNEL_SIZE
        rowIdxEnd   = rowIdxStart + KERNEL_SIZE
        
        for j in range(NUM_KERNELS):
            colIdxStart = j * KERNEL_SIZE
            colIdxEnd   = colIdxStart + KERNEL_SIZE

            img[ rowIdxStart:rowIdxEnd, colIdxStart:colIdxEnd, : ] = block


    print("The dummy image is ")
    print(img)

    # Save the dummy image.
    cv2.imwrite("DummyImage_TestAverageColorValues.bmp", img)

    # The average color values for each channel of the block.
    for k in range(3):
        print("Average value for channel %d is %f." % (k, block[:, :, k].mean() ))
