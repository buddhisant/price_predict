#include <cuda.h>

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N)
{
    int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int max_block_num = 4096;
    return min(optimal_block_num, max_block_num);
}