import numpy as np
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule


#Globals

CUDA_NUM_READS = 1024 * 32

max_threads = 1024  # MAX_THREADS_PER_BLOCK on GeForce 750i
#TODO: figure out how to figure out max threads for any device

def encode_lmer_device(buffer, bufferSize, readCount, readLength, d_lmers, lmerLength, entriesCount):
# def encode_lmer_device(readBuffer, bufferSize, readCount, readLength, d_lmers, lmerLength):
    mod = SourceModule(
        """
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

    __global__ void encodeLmerDevice(
            char  * buffer,
        //	const unsigned int buffSize,
        //	const unsigned int readLength,
            KEY_PTR lmers,
            const unsigned int lmerLength
            )
    {
        __syncthreads();
        printf("in GPU ");
        extern __shared__ char dnaRead[]; // MB: changed from 'read' to solve compile error
        const unsigned int tid=threadIdx.x;
        const unsigned int rOffset=(blockDim.x*blockDim.y*gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x) + (blockDim.x*threadIdx.y);
        KEY_T lmer=0;

        dnaRead[tid]=buffer[rOffset+tid];
        __syncthreads();

        for (unsigned int i = 0; i < 8; i++)    //calculate lmer
        {
            lmer= (lmer<< 8) |	((KEY_T)(shifter[codeF[dnaRead[threadIdx.x+i*4]& 0x07]][3] |
                                shifter[codeF[dnaRead[threadIdx.x+i*4+1]& 0x07]][2] |
                                shifter[codeF[dnaRead[threadIdx.x+i*4+2]& 0x07]][1] |
                                codeF[dnaRead[threadIdx.x+i*4+3] & 0x07]) ) ;
        }
        lmer = (lmer >> ((32 - lmerLength) << 1)) & lmerMask[lmerLength-1];
        printf("%llu", lmer);
        lmers[rOffset+tid]=lmer;


    }
    """
    )
    CUDA_NUM_READS = 1024 * 32
    if readCount < CUDA_NUM_READS:
        readToProcess = readCount
    else:
        readToProcess = CUDA_NUM_READS


    # d_buffer = np.array(buffer, dtype=str)
    gpu_buffer = drv.mem_alloc(buffer.size * buffer.dtype.itemsize)


    encode_lmer = mod.get_function("encodeLmerDevice")
    # encode_lmer.prepare("PPii")

    # lmers = np.empty_like(d_lmers)
    gpu_lmers = drv.mem_alloc(d_lmers.size * d_lmers.dtype.itemsize)


#     grid = (1,1,1)
#     block = (1,1,1)
#     params = {}
# # getOptimalLaunchConfigCustomized(entriesCount,&grid,&block,readLength);
#     getOptimalLaunchConfiguration(entriesCount, grid, block, params)

    block = (max_threads, 1, 1)
    grid_d= {'x':1, 'y': 1, 'z':1}

# if (threadCount > block->x) {
# 		grid->y = threadCount / (block->x);
# 		if (threadCount % (block->x) > 0)
# 			grid->y++;
# 		grid->x = grid->y / 65535 + 1;
# 		grid->y = (grid->y > 65535 ) ? 65535 : grid->y;
# 		grid->z = 1;
# 	}

    if entriesCount > max_threads:
        grid_d['y'] = entriesCount // max_threads
        if entriesCount % max_threads > 0:
            grid_d['y'] += 1
        grid_d['x'] = grid_d['y'] // 65535 + 1
        if grid_d['y'] > 65535:
            grid_d['y'] = 65535
    # grid = (grid_d['x'],grid_d['y'], grid_d['z'])
    grid = (grid_d['x'], grid_d['y'], grid_d['z'])
    np_lmerLength = np.uint64(lmerLength)
    gpu_lmerLength = drv.mem_alloc(np_lmerLength.size * np_lmerLength.dtype.itemsize)
    drv.memcpy_htod(gpu_buffer, buffer)
    drv.memcpy_htod(gpu_lmers,d_lmers)
    drv.memcpy_htod(gpu_lmerLength, np_lmerLength)
    # encode_lmer(gpu_buffer, gpu_lmers, gpu_lmerLength, block=block, grid=grid, shared=128)
    encode_lmer(drv.In(buffer), drv.Out(d_lmers), gpu_lmerLength, block=block, grid=grid, shared=readLength+31)
    drv.memcpy_dtoh(d_lmers, gpu_lmers)
    print(d_lmers)


def compute_kmer_device(lmers):

    mod = SourceModule("""
    __global__ void computeKmerDevice(
            KEY_PTR lmers,
            KEY_PTR pkmers,
            KEY_PTR skmers,
            KEY_T validBitMask
        )
    {

        const unsigned int tid=(blockDim.x*blockDim.y*gridDim.x*blockIdx.y) +(blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
        KEY_T lmer;
        //fetch lmer
        lmer=lmers[tid];
        //find prefix
        pkmers[tid]=LMER_PREFIX(lmer,validBitMask);
        //find suffix
        skmers[tid] = LMER_SUFFIX(lmer,validBitMask);
    }
    """)
    compute_kmer = mod.get_function("computeKmerDevice")
    pkmers = np.zeroes_like(lmers)
    skmers = np.zeroes_like(lmers)

    compute_kmer(
        drv.In(lmers), drv.Out(pkmers), drv.Out(skmers),
        block=(), grid=()  # TODO: figure out block & grid values
        )
    return pkmers, skmers




def compute_lmer_complement_device(readBuffer, bufferSize, readCount, readLength, d_lmers, lmerLength, entriesCount):

    mod = SourceModule("""
    __global__ void encodeLmerComplementDevice(
            char  * buffer,
            const unsigned int buffSize,
            const unsigned int readLength,
            KEY_PTR lmers,
            const unsigned int lmerLength
            )
    {

        extern __shared__ char dnaRead[];//have to fix it
        const unsigned int tid=threadIdx.x;
        const unsigned int rOffset=(blockDim.x*blockDim.y*gridDim.x*blockIdx.y) +(blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y);
        KEY_T lmer=0;
        KEY_T temp=0;

        dnaRead[tid]=buffer[rOffset+tid];
        __syncthreads();
        dnaRead[tid]=codeR[dnaRead[tid] & 0x07];
        __syncthreads();
        for(unsigned int i =0; i< lmerLength; i++)
        {
            temp=((KEY_T)dnaRead[(tid+i)%blockDim.x]);
            lmer = (temp<<(i<<1)) | lmer;
        }
        lmers[rOffset+tid]=lmer;

    }
    """)

    encode_lmer_complement = mod.get_function("encodeLmerComplementDevice")
    if readCount < CUDA_NUM_READS:
        readToProcess = readCount
    else:
        readToProcess = CUDA_NUM_READS
    kmerBitMask = 0

    buffer = np.fromstring(''.join(readBuffer), dtype='uint8')

    lmers = np.zeros(len(readBuffer), dtype='uint64')

    kmerBitMask = 0
    for _ in range(0, (lmerLength - 1) * 2):
        kmerBitMask = (kmerBitMask << 1) | 1

    encode_lmer_complement(
        drv.In(buffer), drv.In(bufferSize), drv.In(readLength), drv.InOut(lmers), drv.In(lmerLength),
        block=(), grid=()  # TODO: figure out block & grid values
    )

    return lmers

# getOptimalLaunchConfigCustomized(entriesCount,&grid,&block,readLength);

def getOptimalLaunchConfiguration(threadCount,gridx, gridy, params):
    """

    :param threadCount:
    :param gridx: 3-tuple representing block size x,y,z
    :param gridy: 3-tuple representing grid size x,y,z
    :param params: dictionary {'threads' : <integer>}
    :return:
    """
    # gridx = {'x':1,'y':1,'z':1}
    # gridy = {'x':1,'y':1,'z':1}
    params['threads'] = 32
    gridy[0] = threadCount // params['threads']
    if threadCount % params['threads'] > 0:
        gridy[1] += 1

# extern "C"
# void getOptimalLaunchConfiguration(
#       unsigned int threadCount,
#       unsigned int * gridx,
#       unsigned int * gridy,
#       unsigned int * threads)
# {
#
#     *threads = 32;
#     *gridy = threadCount / (*threads);
#     if (threadCount % (*threads) > 0)(*gridy) + +;
#     (*gridx) = (*gridy) / 65535 + 1;
#     (*gridy) = (*gridy) % 65535;
# }
