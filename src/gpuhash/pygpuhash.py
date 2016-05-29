""" PyCUDA implementation of GPUHash """
import sys
import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.scan import ExclusiveScanKernel
import logging


module_logger = logging.getLogger('eulercuda.pygpuhash')

MAX_BUCKET_ITEM = 520


def phase1_device(d_keys, d_offset, d_length, count, bucketCount):
    logger = logging.getLogger('eulercuda.pygpuhash.phase1_device')
    logger.info("started.")
    mod = SourceModule("""
    #include <stdio.h>
    typedef unsigned long long  KEY_T ;
    typedef KEY_T               *KEY_PTR;
    typedef unsigned int        VALUE_T;
    typedef VALUE_T             *VALUE_PTR;
    #define C0  0x01010101
    #define C1	0x12345678
    #define LARGE_PRIME 1900813
    #define MAX_INT  0xffffffff

    __forceinline__ __device__ unsigned int hash_h(KEY_T  key, unsigned int bucketCount)
    {
        return ((C0 + C1 * key) % LARGE_PRIME ) % bucketCount;
    }
    __global__ void phase1(	KEY_PTR  keys,
                unsigned int * offset,
                unsigned int length,
                unsigned int* count,
                unsigned int bucketCount)
    {

        unsigned int tid = (blockDim.x * blockDim.y * gridDim.x * blockIdx.y) +
        (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y) + threadIdx.x;
        if(tid < length)
        {
            KEY_T key = keys[tid];
            unsigned int bucket = hash_h(key,bucketCount);
            offset[tid] = atomicInc( count + bucket, MAX_INT);
            printf(" offset = %u ", offset[tid]);

        }
        __syncthreads();
    }
    """, keep = True)
    np_d_keys = np.array(d_keys, dtype = np.uint64)
    # np_d_offset = np.zeros(d_offset, dtype = np.uint)

    block_dim = (512, 1, 1)
    if (d_length//512) == 0:
        grid_dim = (1, 1, 1)
    else:
        grid_dim = (d_length//512, 1, 1)
    phase1 = mod.get_function("phase1")
    phase1(
        drv.InOut(np_d_keys),
        drv.InOut(d_offset),
        np.uint(d_length),
        drv.Out(count),
        np.uint(bucketCount),
        grid = grid_dim,
        block = block_dim
    )
    # d_bucketSize = np_d_bucketSize.tolist()
    # d_offset = np_d_offset.tolist()
    logger.info('Finished. Leaving.')
 #   return [d_offset, d_bucketSize]
    return d_offset, count


def copy_to_bucket_device(d_keys, d_values, d_offset, d_length, d_start, bucketCount, d_bufferK, d_bufferV):
    logger = logging.getLogger('eulercuda.pygpuhash.copy_to_bucket_device')
    logger.info("started.")
    mod = SourceModule("""
    #include <stdio.h>
    typedef unsigned long long  KEY_T ;
    typedef KEY_T               *KEY_PTR;
    typedef unsigned int        VALUE_T;
    typedef VALUE_T             *VALUE_PTR;
    #define C0  0x01010101
    #define C1	0x12345678
    #define LARGE_PRIME 1900813
    #define MAX_INT  0xffffffff

    __forceinline__ __device__ unsigned int hash_h(KEY_T  key, unsigned int bucketCount)
    {
        return ((C0 + C1 * key) % LARGE_PRIME ) % bucketCount;
    }
     __global__ void copyToBucket(	KEY_PTR keys,
                    VALUE_PTR values,
                    unsigned int * offset,
                    unsigned int length,
                    unsigned int* start,
                    unsigned int bucketCount,
                    KEY_PTR  bufferK,
                    VALUE_PTR bufferV)
    {

        unsigned tid = (blockDim.x * blockDim.y * gridDim.x * blockIdx.y) +
                    (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y) + threadIdx.x;

        if (tid < length)
      {
            KEY_T key = keys[tid];
            unsigned int bucket = hash_h(key,bucketCount);
            VALUE_T value = values[tid];
            unsigned int index = start[bucket] + offset[tid];
            printf(" index = %d ", index);
            bufferK[index] = key;
            bufferV[index] = value;

        }
    }
    """, keep = True)
    copy_to_bucket = mod.get_function("copyToBucket")

    d_keys = np.array(d_keys, dtype = np.uint64)
    d_values = np.array(d_values, dtype = np.uint)
    # d_offset = np.array(d_values, dtype = np.uint)
    # d_start = np.array(d_start, dtype = np.uint)
    d_bufferK = np.zeros(d_keys.size, dtype = np.uint64)
    d_bufferV = np.zeros(d_values.size, dtype = np.uint)
    block_dim = (512, 1, 1)
    if (d_length//512) == 0:
        grid_dim = (1, 1, 1)
    else:
        grid_dim = (d_length//512, 1, 1)
    copy_to_bucket(
        drv.In(d_keys),
        drv.In(d_values),
        drv.In(d_offset),
        np.uint(d_length),
        drv.In(d_start),
        np.uint(bucketCount),
        drv.Out(d_bufferK),
        drv.Out(d_bufferV),
        grid = grid_dim,
        block = block_dim
    )
    logger.info('Finished. Leaving.')
    return [d_bufferK, d_bufferV]


def bucket_sort_device(d_bufferK, d_bufferV, d_start, d_bucketSize, bucketCount, d_TK, d_TV):
    logger = logging.getLogger('eulercuda.pygpuhash.bucket_sort_device')
    logger.info("started.")
    mod = SourceModule("""
    typedef unsigned long long  KEY_T ;
    typedef KEY_T               *KEY_PTR;
    typedef unsigned int        VALUE_T;
    typedef VALUE_T             *VALUE_PTR;
    #define MAX_BUCKET_ITEM (520)

    #define GET_KEY_INDEX(blockIdx,itemIdx) ((blockIdx)*MAX_BUCKET_ITEM+(itemIdx))
    #define GET_VALUE_INDEX(blockIdx,itemIdx) ((blockIdx)*MAX_BUCKET_ITEM+(itemIdx))

    __global__ void bucketSort(KEY_PTR bufferK, VALUE_PTR bufferV, unsigned int *start, unsigned int *bucketSize,
                unsigned int bucketCount, KEY_PTR TK, VALUE_PTR TV)
    {

        __shared__ KEY_T keys[MAX_BUCKET_ITEM];
        unsigned int keyCount[MAX_BUCKET_ITEM / 32];

        unsigned int blockOffset = start[blockIdx.x];
        unsigned int size = bucketSize[blockIdx.x];

        unsigned int chunks = size >> 5;
        chunks = (chunks << 5 == size) ? chunks : chunks + 1;
        for(unsigned int j = 0; j < chunks; j++)
        {
            if ((j << 5) + threadIdx.x < size)
                keys[(j << 5) + threadIdx.x] = bufferK[blockOffset + (j << 5) + threadIdx.x];
        }

        __syncthreads();
        for(unsigned int j = 0;j < chunks; j++)
        {
            if((j << 5) + threadIdx.x < size)
            {
                keyCount[j] = 0;
                for(int i=0; i < size; i++)
                {
                    keyCount[j] = ( keys[(j << 5) + threadIdx.x] > keys[i] ) ? keyCount[j] + 1 : keyCount[j];
                }
            }
        }
            __syncthreads();
        for(unsigned int j = 0; j < chunks; j++)
        {
            if((j << 5) + threadIdx.x < size)
            {
                TK[GET_KEY_INDEX(blockIdx.x, keyCount[j])] = keys[(j << 5) + threadIdx.x];
                TV[GET_VALUE_INDEX(blockIdx.x, keyCount[j])] = bufferV[blockOffset + (j << 5) + threadIdx.x];
            }
        }
    }
    """)
    bucket_sort = mod.get_function('bucketSort')
    # TODO: Figure out why d_TK and d_TV come out with only 1 element each.
    d_TV = np.empty(d_bufferV.size, dtype = np.uint64)
    d_TK = np.empty(d_bufferK.size, dtype = np.uint64)
    block_dim = (bucketCount, 1, 1)
    grid_dim = (32, 1, 1)
    bucket_sort(
        drv.In(d_bufferK),
        drv.In(d_bufferV),
        drv.In(d_start),
        drv.In(d_bucketSize),
        np.uint(bucketCount),
        drv.InOut(d_TK),
        drv.InOut(d_TV),
        grid = grid_dim,
        block = block_dim # What about shared? Original source doesn't have it.

    )
    logger.info("Finished. Leaving.")
    return [d_TK, d_TV]


def create_hash_table_device(d_keys, d_values, d_length, d_TK, d_TV, tableLength, d_bucketSize, bucketCount):
    logger = logging.getLogger('eulercuda.pygpuhash.create_hash_table_device')
    logger.info("started.")
    d_offset = 0
    d_start = 0

    d_bufferK = []
    d_bufferV = []
    bucketCount = (d_length // 409) + 1 # ceil

    dataSize = d_length * sys.getsizeof(np.uint)
    bucketDataSize = bucketCount * sys.getsizeof(np.uint)
    d_bucketSize = np.zeros(len(d_keys), dtype = np.uint)
    d_offset = np.empty(dataSize, dtype = np.uint)
    #   d_bucketSize = np.empty(bucketDataSize, dtype = np.uint)
    #   think d_bucketSize needs to be a 2D array
    result = phase1_device(d_keys, d_offset, d_length, d_bucketSize, bucketCount)
    d_offset, d_bucketSize = result

    d_start = np.empty(len(d_keys), dtype = np.uint)

    # cudppScan(scanplan, d_start, *d_bucketSize, *bucketCount);
    # cudppScan (const CUDPPHandle planHandle, void *d_out, const void *d_in, size_t numElements)

    knl = ExclusiveScanKernel(np.uint, "a+b", 0)
    # flat_bucketsize = np.array(d_bucketSize.flatten())
    np_d_start = gpuarray.to_gpu(d_bucketSize)
    logger.info('Prior to scan.')
    knl(np_d_start)
    logger.info('Post scan.')
    d_start = np_d_start.get()
    # / ** ** ** ** ** ** ** *Cuckoo Hashing ** ** ** ** ** ** ** ** ** /
    result = copy_to_bucket_device(d_keys, d_values, d_offset, d_length, d_start, bucketCount, d_bufferK, d_bufferV)
    d_bufferK = result[0]
    d_bufferV = result[1]

    result = bucket_sort_device(d_bufferK, d_bufferV, d_start, d_bucketSize, bucketCount, d_TK, d_TV)
    tableLength = bucketCount * MAX_BUCKET_ITEM

    return [tableLength, d_bucketSize, bucketCount, result[0], result[1]]