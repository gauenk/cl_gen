
// histogramGPU computes the histogram of an input array on the GPU
__global__ void histogramGPU(unsigned int* input, unsigned int* bins, unsigned int numElems) {
    int tx = threadIdx.x; int bx = blockIdx.x;

    // compute global thread coordinates
    int i = (bx * blockDim.x) + tx;

    // create a private histogram copy for each thread block
    __shared__ unsigned int hist[PRIVATE];

    // each thread must initialize more than 1 location
    if (PRIVATE > BLOCK_SIZE) {
        for (int j=tx; j<PRIVATE; j+=BLOCK_SIZE) {
            if (j < PRIVATE) {
                hist[j] = 0;
            }
        }
    }
    // use the first `PRIVATE` threads of each block to init
    else {
        if (tx < PRIVATE) {
            hist[tx] = 0;
        }
    }
    // wait for all threads in the block to finish
    __syncthreads();

    // update private histogram
    if (i < numElems) {
        atomicAdd(&(hist[input[i]]), 1);
    }
    // wait for all threads in the block to finish
    __syncthreads();

    // each thread must update more than 1 location
    if (PRIVATE > BLOCK_SIZE) {
        for (int j=tx; j<PRIVATE; j+=BLOCK_SIZE) {
            if (j < PRIVATE) {
                atomicAdd(&(bins[j]), hist[j]);
            }
        }
    }
    // use the first `PRIVATE` threads to update final histogram
    else {
        if (tx < PRIVATE) {
            atomicAdd(&(bins[tx]), hist[tx]);
        }
    }