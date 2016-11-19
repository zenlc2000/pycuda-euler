""" PyCUDA implementation of GPUHash """
import sys
import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.scan import ExclusiveScanKernel
import logging
import sys
sys.path.append('/usr/local/cuda-7.5/bin')

module_logger = logging.getLogger('eulercuda.pygpuhash')

MAX_BUCKET_ITEM = 520
ULONGLONG = 8
UINTC = 4

def phase1_device(d_keys, d_offset, d_length, count, bucketCount):
    logger = logging.getLogger('eulercuda.pygpuhash.phase1_device')
    logger.info("started.")
    mod = SourceModule("""
    //#include <stdio.h>
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
                unsigned int bucketCount){

        unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
        if(tid<length)
        {
            KEY_T key=keys[tid];
            unsigned int bucket=hash_h(key,bucketCount);
            offset[tid]=atomicInc (count+bucket,MAX_INT);

        }
        __syncthreads();
    }
    """, options=['--compiler-options', '-Wall'])
    np_d_keys = np.array(d_keys).astype('Q')
    keys_gpu = gpuarray.to_gpu(np_d_keys)
    offset_gpu = gpuarray.zeros(len(d_keys), dtype='I')
    count_gpu = gpuarray.to_gpu(count)
    block_dim = (1024, 1, 1)
    if (d_length//1024) == 0:
        grid_dim = (1, 1, 1)
    else:
        grid_dim = (d_length//1024, 1, 1)
    phase1 = mod.get_function("phase1")
    logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    phase1(keys_gpu, offset_gpu, np.uintc(d_length), count_gpu,
            np.uintc(bucketCount), grid=grid_dim, block=block_dim)
    d_offset = offset_gpu.get()
    count = count_gpu.get()
    devdata = pycuda.tools.DeviceData()
    # orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[0])
    # logger.info("Occupancy = %s" % (orec.occupancy * 100))
    logger.info('Finished. Leaving.')
 #   return [d_offset, d_bucketSize]
    return d_offset, count


def copy_to_bucket_device(d_keys, d_values, d_offset, d_length, d_start, bucketCount, d_bufferK, d_bufferV):
    logger = logging.getLogger('eulercuda.pygpuhash.copy_to_bucket_device')
    logger.info("started.")
    mod = SourceModule("""
   // #include <stdio.h>
    //typedef unsigned long long  KEY_T ;
    //typedef KEY_T               *KEY_PTR;
    //typedef unsigned int        VALUE_T;
    //typedef VALUE_T             *VALUE_PTR;
    #define C0  0x01010101
    #define C1	0x12345678
    #define LARGE_PRIME 1900813
    #define MAX_INT  0xffffffff

    __forceinline__ __device__ unsigned int hash_h(unsigned long long key, unsigned int bucketCount)
    {
        return ((C0 + C1 * key) % LARGE_PRIME ) % bucketCount;
    }
     __global__ void copyToBucket(	unsigned long long *keys,
                    unsigned int *values,
                    unsigned int * offset,
                    unsigned int length,
                    unsigned int* start,
                    unsigned int bucketCount,
                    unsigned long long * bufferK,
                    unsigned int *bufferV)
    {

        unsigned tid = (blockDim.x * blockDim.y * gridDim.x * blockIdx.y) +
                    (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y) + threadIdx.x;

        if (tid < length)
      {
            unsigned long long key = keys[tid];
            unsigned int bucket = hash_h(key,bucketCount);

           // printf(" bucket = %u ", bucket);

            unsigned int value = values[tid];
            unsigned int index = start[bucket] + offset[tid];

           // printf(" index = %u ", index);
           // printf(" tid = %u, offset = %u bucket = %u start = %u index = %u ", tid, offset[tid], bucket, start[bucket], (start[bucket] + offset[tid]));

            bufferK[index] = key;
            bufferV[index] = value;

            //printf(" bufferV = %u ", bufferV[index]);

        }
    }
    """)
    copy_to_bucket = mod.get_function("copyToBucket")

    np_d_keys = np.array(d_keys).astype(np.ulonglong)
    np_d_values = np.array(d_values).astype(np.uintc)
    # np_d_start = np.array(len(d_keys), dtype = np.uint32)
    # np_d_bufferK = np.empty(np_d_keys.size, dtype = np.uint64)
    #
    # np_d_bufferV = np.empty(np_d_values.size, dtype = np.uint32)

    keys_gpu = gpuarray.to_gpu(np_d_keys)
    values_gpu = gpuarray.to_gpu(np_d_values)
    offset_gpu = gpuarray.to_gpu(d_offset)
    # start_gpu = gpuarray.to_gpu(np_d_start)
    np_d_bufferK = gpuarray.to_gpu(d_bufferK)
    np_d_bufferV = gpuarray.to_gpu(d_bufferV)
    block_dim = (1024, 1, 1)
    if (d_length//1024) == 0:
        grid_dim = (1, 1, 1)
    else:
        grid_dim = (d_length//1024, 1, 1)
    logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    copy_to_bucket(
        keys_gpu,
        values_gpu,
        offset_gpu,
        np.uintc(d_length),
        drv.In(d_start), #start_gpu,
        np.uintc(bucketCount),
        np_d_bufferK, #bufferK_gpu,
        np_d_bufferV, #bufferV_gpu,
        grid = grid_dim,
        block = block_dim
    )
    # devdata = pycuda.tools.DeviceData()
    # orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[0])
    # logger.info("Occupancy = %s" % (orec.occupancy * 100))
    np_d_bufferK.get(d_bufferK)
    np_d_bufferV.get(d_bufferV)
    logger.info('Finished. Leaving.')
    # d_start  = start_gpu.get()
    # d_bufferK = bufferK_gpu.get()
    # d_bufferV = bufferV_gpu.get()
    return d_bufferK, d_bufferV


def bucket_sort_device(d_bufferK, d_bufferV, d_start, d_bucketSize, bucketCount, d_TK, d_TV):
    logger = logging.getLogger('eulercuda.pygpuhash.bucket_sort_device')
    logger.info("started.")
    mod = SourceModule("""
    typedef unsigned long long  KEY_T ;
    typedef KEY_T               *KEY_PTR;
    typedef unsigned int        VALUE_T;
    typedef VALUE_T             *VALUE_PTR;
    #define MAX_BUCKET_ITEM (520)

    #define GET_KEY_INDEX(blockIdx,itemIdx) ((blockIdx) * MAX_BUCKET_ITEM + (itemIdx))
    #define GET_VALUE_INDEX(blockIdx,itemIdx) ((blockIdx) * MAX_BUCKET_ITEM + (itemIdx))

    __global__ void bucketSort(
                                KEY_PTR         bufferK,
                                VALUE_PTR       bufferV,
                                unsigned int    *start,
                                unsigned int    *bucketSize,
                                unsigned int    bucketCount,
                                KEY_PTR         TK,
                                VALUE_PTR       TV)
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
        for (unsigned int j = 0; j < chunks; j++)
        {
            if ((j << 5) + threadIdx.x < size)
            {
                keyCount[j] = 0;
                for(int i=0; i < size; i++)
                {
                    keyCount[j] = ( keys[(j << 5) + threadIdx.x] > keys[i] ) ? keyCount[j] + 1 : keyCount[j];
                }
            }
        }
            __syncthreads();
        for (unsigned int j = 0; j < chunks; j++)
        {
            if ((j << 5) + threadIdx.x < size)
            {
                TK[GET_KEY_INDEX(blockIdx.x, keyCount[j])] = keys[(j << 5) + threadIdx.x];
                TV[GET_VALUE_INDEX(blockIdx.x, keyCount[j])] = bufferV[blockOffset + (j << 5) + threadIdx.x];
            }
        }
    }
    """)
    bucket_sort = mod.get_function('bucketSort')
    block_dim = (32, 1, 1)
    grid_dim = (bucketCount, 1, 1)#(32, 1, 1)
    logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    np_d_TK = gpuarray.to_gpu(d_TK)
    np_d_TV = gpuarray.to_gpu(d_TV)
    bucket_sort(
        drv.In(d_bufferK),
        drv.In(d_bufferV),
        drv.In(d_start),
        drv.In(d_bucketSize),
        np.uintc(bucketCount),
        np_d_TK,
        np_d_TV,
        grid=grid_dim,
        block=block_dim # What about shared? Original source doesn't have it.

    )
    np_d_TK.get(d_TK)
    np_d_TV.get(d_TV)
    devdata = pycuda.tools.DeviceData()
    # orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[0])
    # logger.info("Occupancy = %s" % (orec.occupancy * 100))

    logger.info("Finished. Leaving.")
    return d_TK, d_TV


def create_hash_table_device(d_keys, d_values, d_length, d_TK, d_TV, tableLength, d_bucketSize, bucketCount):
    logger = logging.getLogger('eulercuda.pygpuhash.create_hash_table_device')
    logger.info("started.")
    # d_start = np.zeros(bucketDataSize, dtype = np.uint32)
    KEY_SIZE = ULONGLONG #sys.getsizeof(np.ulonglong)
    VALUE_SIZE = UINTC #sys.getsizeof(np.uintc)

    # BUCKET_KEY_SIZE = KEY_SIZE * MAX_BUCKET_ITEM
    # BUCKET_VALUE_SIZE = VALUE_SIZE * MAX_BUCKET_ITEM
    BUCKET_KEY_SIZE = MAX_BUCKET_ITEM
    BUCKET_VALUE_SIZE = MAX_BUCKET_ITEM

    bucketCount = (d_length // 409) + 1 # ceil

    dataSize = d_length #* UINTC# .getsizeof(np.uintc)
    bucketDataSize = bucketCount #* UINTC #  sys.getsizeof(np.uintc)
    d_bucketSize = np.zeros(bucketCount, dtype='I')
    # d_bucketSize[0] = bucketDataSize
    d_offset = np.zeros(dataSize, dtype='I')
    #   d_bucketSize = np.empty(bucketDataSize, dtype = np.uint)
    #   think d_bucketSize needs to be a 2D array
    # result = \

# launch phase 1 , bucket allocation
    d_offset, d_bucketSize = phase1_device(d_keys, d_offset, d_length, d_bucketSize, bucketCount)

#/************  Calculating Start of each bucket (prefix sum of Count) **********/
    d_start = np.zeros(bucketDataSize, dtype='I')
    knl = ExclusiveScanKernel(np.uintc, "a+b", 0)
    # flat_bucketsize = np.array(d_bucketSize.flatten())
    np_d_start = gpuarray.to_gpu(d_bucketSize)
    logger.info('Prior to scan.')
    knl(np_d_start)
    logger.info('Post scan.')
    d_start = np_d_start.get()

    d_bufferK = np.zeros(d_length).astype(np.ulonglong)
    d_bufferV = np.zeros(d_length).astype(np.uintc)
    # / ** ** ** ** ** ** ** *Cuckoo Hashing ** ** ** ** ** ** ** ** ** /
    # result = \
    d_bufferK, d_bufferV = copy_to_bucket_device(d_keys, d_values, d_offset, d_length, d_start, bucketCount, d_bufferK, d_bufferV)
    # d_bufferK = result[0]
    # d_bufferV = result[1]
    # d_start = result[2]
    d_TK = np.zeros(bucketCount * MAX_BUCKET_ITEM, dtype='Q')
    d_TV = np.zeros(bucketCount * MAX_BUCKET_ITEM, dtype='I')
    # result = \
    d_TK, d_TV = bucket_sort_device(d_bufferK, d_bufferV, d_start, d_bucketSize, bucketCount, d_TK, d_TV)
    with open ('hash_tk.txt', 'w') as ofile:
        for line in d_TK:
            ofile.write(str(line) + '\t' )
    tableLength = bucketCount * MAX_BUCKET_ITEM
    logger.info("Finished. Leaving.")
    return [tableLength, d_bucketSize, bucketCount, d_TK, d_TV]
