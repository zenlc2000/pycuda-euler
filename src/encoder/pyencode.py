import numpy as np
import pycuda.driver as drv
import pycuda.autoinit as autoinit
from pycuda.compiler import SourceModule
import logging
import pycuda
import pycuda.gpuarray as gpuarray
from pycuda.tools import OccupancyRecord
module_logger = logging.getLogger('eulercuda.pyencode')

# ULONGLONG = 8
# UINTC = 4

def encode_lmer_device (buffer, readCount, d_lmers, readLength, lmerLength):
    # module_logger = logging.getLogger('eulercuda.pyencode.encode_lmer_device')
    module_logger.info("started encode_lmer_device.")
    # readLength is total number of bases read.
    mod = SourceModule("""
    #include <stdio.h>
    typedef unsigned  long long KEY_T ;
    typedef KEY_T * KEY_PTR ;
    __device__ __constant__ KEY_T lmerMask[] ={
    0x0000000000000003, 0x000000000000000F, 0x000000000000003F, 0x00000000000000FF, // 0   1   2   3
    0x00000000000003FF, 0x0000000000000FFF, 0x0000000000003FFF, 0x000000000000FFFF, // 4   5   6   7
    0x000000000003FFFF, 0x00000000000FFFFF, 0x00000000003FFFFF, 0x0000000000FFFFFF, // 8   9   10  11
    0x0000000003FFFFFF, 0x000000000FFFFFFF, 0x000000003FFFFFFF, 0x00000000FFFFFFFF, // 12  13  14  15
    0x00000003FFFFFFFF, 0x0000000FFFFFFFFF, 0x0000003FFFFFFFFF, 0x000000FFFFFFFFFF, // 16  17  18  19
    0x000003FFFFFFFFFF, 0x00000FFFFFFFFFFF, 0x00003FFFFFFFFFFF, 0x0000FFFFFFFFFFFF, // 20  21  22  23
    0x0003FFFFFFFFFFFF, 0x000FFFFFFFFFFFFF, 0x003FFFFFFFFFFFFF, 0x00FFFFFFFFFFFFFF, // 24  25  26  27
    0x03FFFFFFFFFFFFFF, 0x0FFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF // 28  29  30  31
    };

    __device__ __constant__ unsigned char shifter[4] [4]=
    {
            {0,0,0,0},
            {1,4,16,64},
            {2,8,32,128},
            {3,12,48,192},
    };
    __device__ __constant__ char  codeF[]={0,0,0,1,3,0,0,2};
    __device__ __constant__ char  codeR[]={0,3,0,2,0,0,0,1};

    __global__ void encodeLmerDevice(	char  * buffer,
                //    const unsigned int buffSize,
                //    const unsigned int readLength,
                    KEY_PTR lmers,
                    const unsigned int lmerLength
                    )
    {

        extern __shared__ char read[];
        const unsigned int tid=threadIdx.x;
        const unsigned int rOffset=(blockDim.x*blockDim.y*gridDim.x*blockIdx.y) +(blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y);
        KEY_T lmer=0;

        read[tid] = buffer[rOffset + tid];

        __syncthreads();

        for (unsigned int i = 0; i < 8; i++)    //calculate lmer
        {
            lmer = (lmer<< 8) |	((KEY_T)(shifter[codeF[read[threadIdx.x+i*4]& 0x07]][3] |
                                    shifter[codeF[read[threadIdx.x+i*4+1]& 0x07]][2] |
                                    shifter[codeF[read[threadIdx.x+i*4+2]& 0x07]][1] |
                                    codeF[read[threadIdx.x+i*4+3] & 0x07]) ) ;
        }
        lmer = (lmer >> ((32 - lmerLength) << 1)) & lmerMask[lmerLength-1];
        // printf(" offset = %u, lmer = %llu ", (tid + rOffset),lmer);
        lmers[rOffset + tid] = lmer;

    }
    """)

    encode_lmer = mod.get_function("encodeLmerDevice")

    block_dim, grid_dim = getOptimalLaunchConfiguration(readCount, readLength)
    module_logger.debug("block_dim = %s, grid_dim = %s" % (block_dim, grid_dim))
    if isinstance(buffer, np.ndarray) and isinstance(d_lmers, np.ndarray):
        module_logger.info("Going to GPU.")
        np_d_lmers = gpuarray.to_gpu(d_lmers)
        encode_lmer(drv.In(buffer),
                    np_d_lmers,
                    np.uintc(lmerLength),
                    block=block_dim,
                    grid=grid_dim,
                    shared=readLength + 31)
        np_d_lmers.get(d_lmers)
    else:
        print(isinstance(buffer, np.ndarray), isinstance(d_lmers, np.ndarray))
    module_logger.debug("Generated %s lmers." % (len(d_lmers)))
    devdata = pycuda.tools.DeviceData()
    orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[0])
    module_logger.debug("Occupancy = %s" % (orec.occupancy * 100))
    module_logger.info("finished encode_lmer_device.")
    return d_lmers


def compute_kmer_device (lmers, pkmers, skmers, kmerBitMask, readLength, readCount):
    # module_logger = logging.getLogger('eulercuda.pyencode.compute_kmer_device')
    module_logger.info("started compute_kmer_device.")
    mod = SourceModule("""
    typedef unsigned  long long KEY_T ;
    typedef KEY_T * KEY_PTR ;
    #define LMER_PREFIX(lmer,bitMask) ((lmer & (bitMask<<2))>>2)
    #define LMER_SUFFIX(lmer,bitMask) ((lmer & bitMask))

    __global__ void computeKmerDevice(
            KEY_PTR lmers,
            KEY_PTR pkmers,
            KEY_PTR skmers,
            KEY_T validBitMask,
            unsigned int readCount
        )
    {
       const unsigned int tid = (blockDim.x * blockDim.y * gridDim.x * blockIdx.y) + (blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * threadIdx.y) + threadIdx.x;
        
        if (tid < readCount)
        {
     

            KEY_T lmer;
            //fetch lmer
            lmer = lmers[tid];
            //find prefix
            pkmers[tid] = LMER_PREFIX(lmer,validBitMask);
            //find suffix
            skmers[tid] = LMER_SUFFIX(lmer,validBitMask);
           // __syncthreads();
        }
    }
    """, options=['--compiler-options', '-Wall'])
    compute_kmer = mod.get_function("computeKmerDevice")

    block_dim, grid_dim = getOptimalLaunchConfiguration(readCount, readLength)
    np_pkmers = gpuarray.to_gpu(pkmers)
    np_skmers = gpuarray.to_gpu(skmers)
    if isinstance(lmers, np.ndarray) and isinstance(pkmers, np.ndarray) and isinstance(skmers, np.ndarray):
        module_logger.info("Going to GPU.")
        compute_kmer(
            drv.In(lmers),
            np_pkmers,
            np_skmers,
            np.ulonglong(kmerBitMask),
            np.uintc(readCount),
            block=block_dim, grid=grid_dim
        )
        np_pkmers.get(pkmers)
        np_skmers.get(skmers)
    else:
        module_logger.warn("PROBLEM WITH GPU.")
    devdata = pycuda.tools.DeviceData()
    orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[0])
    module_logger.debug("Occupancy = %s" % (orec.occupancy * 100))

    module_logger.info("leaving compute_kmer_device.")
    return pkmers, skmers


def compute_lmer_complement_device(buffer, readCount, d_lmers, readLength, lmerLength):
    # logger = logging.getLogger('eulercuda.pyencode.compute_lmer_complement_device')
    module_logger.info("started compute_lmer_complement_device.")
    mod = SourceModule("""
    __device__ __constant__ char  codeF[]={0,0,0,1,3,0,0,2};
    __device__ __constant__ char  codeR[]={0,3,0,2,0,0,0,1};
    typedef unsigned  long long KEY_T ;
    typedef KEY_T * KEY_PTR ;

    __global__ void encodeLmerComplementDevice(
            char  * buffer,
            KEY_PTR lmers,
            const unsigned int lmerLength,
            const unsigned int readCount
            )
    {
        const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
     //   const unsigned int col = blockIdx.y + blockDim.y + threadIdx.y;

        
        if (tid < readCount)
        {
     
            extern __shared__ char dnaRead[];
            //unsigned int lmerLength = 0;
            KEY_T lmer = 0;
            KEY_T temp = 0;
    
           // lmerLength = d_lmerLength[tid];
            dnaRead[tid] = buffer[row + tid];
    
            __syncthreads();
    
            dnaRead[tid] = codeR[dnaRead[tid] & 0x07];
            __syncthreads();
    
            for (unsigned int i = 0; i < lmerLength; i++)
            {
                temp = ((KEY_T)dnaRead[(tid + i) % blockDim.x]);
                lmer = (temp << (i << 1)) | lmer;
            }
            lmers[row + tid] = lmer;
            __syncthreads();
        }
    }
    """, options=['--compiler-options', '-Wall'])

    encode_lmer_complement = mod.get_function("encodeLmerComplementDevice")
    block_dim, grid_dim = getOptimalLaunchConfiguration(readCount, readLength)

    module_logger.debug('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    if isinstance(buffer, np.ndarray) and isinstance(d_lmers, np.ndarray):
        np_lmerLength = np.uintc(lmerLength)
        np_d_lmers = gpuarray.to_gpu(d_lmers)
        module_logger.info("Going to GPU.")
        encode_lmer_complement(
            drv.In(buffer),  np_d_lmers, np_lmerLength, np.uintc(readCount),
            block=block_dim, grid=grid_dim,  shared=readLength + 31
        )
        np_d_lmers.get(d_lmers)
    else:
        print("Problem with data to GPU")
        module_logger.warn("problem with data to GPU.")

    devdata = pycuda.tools.DeviceData()
    orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[0])
    module_logger.info("Occupancy = %s" % (orec.occupancy * 100))

    module_logger.info("Finished compute_lmer_complement_device.")
    return d_lmers


# getOptimalLaunchConfigCustomized(entriesCount,&grid,&block,readLength);

def getOptimalLaunchConfiguration (threadCount, threadPerBlock):
    """
    :param threadCount:
    :param threadPerBlock:
    :return: block_dim, grid_dim - 3-tuples for block and grid x, y, and z
    """
    block = {'x': threadPerBlock, 'y': 1, 'z': 1}
    grid = {'x': 1, 'y': 1, 'z': 1}

    if threadCount > block['x']:
        grid['y'] = threadCount // block['x']
        if threadCount % block['x'] > 0:
            grid['y'] += 1
        grid['x'] = grid['y'] // 65535 + 1
        if grid['y'] > 65535:
            grid['y'] = 65535
    block_dim = (block['x'], block['y'], block['z'])
    grid_dim = (grid['x'], grid['y'], grid['z'])
    return block_dim, grid_dim
# 	getOptimalLaunchConfigCustomized(entriesCount,&grid,&block,readLength);
#
