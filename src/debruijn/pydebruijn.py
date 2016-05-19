import sys
import numpy as np
import pycuda.driver as drv
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.scan import ExclusiveScanKernel
from pycuda.driver import device_attribute
import logging
from encoder.pyencode import getOptimalLaunchConfiguration

module_logger = logging.getLogger('eulercuda.pydebruijn')


def debruijn_count_device(d_lmerKeys, d_lmerValues, lmerCount, d_TK, d_TV, d_bucketSeed, bucketCount,
                          d_lcount, d_ecount, valid_bitmask):
    """

    This kernel works on each l-mer ,counting edges of the graph.


    :return:
    """
    logger = logging.getLogger('eulercuda.pydebruijn.debruijn_count')


    logger.info("started.")
    mod = SourceModule("""
    typedef unsigned long long  KEY_T ;
    typedef KEY_T               *KEY_PTR;
    typedef unsigned int        VALUE_T;
    typedef VALUE_T             *VALUE_PTR;

    #define LARGE_PRIME 1900813
    #define L2_SIZE 192
    #define MAX_ITERATIONS 100
    #define MAX_INT 0xffffffff
    #define MAX_SEED_COUNT 25
    #define C0  0x01010101
    #define C1	0x12345678
    #define C10 0xABCDEFAB
    #define C11 0xCDEFABCD
    #define C20 0xEFABCDEF
    #define C21 0xBAFEDCBA
    #define C30 0xFEDCBAFE
    #define C31 0xDCBAFEDC
    #define GET_KEY_INDEX(blockIdx,itemIdx) ((blockIdx)*MAX_BUCKET_ITEM+(itemIdx))
    #define GET_VALUE_INDEX(blockIdx,itemIdx) ((blockIdx)*MAX_BUCKET_ITEM+(itemIdx))
    #define MAX_BUCKET_ITEM (520)


    __forceinline__ __device__ unsigned int hash_h(KEY_T  key, unsigned int bucketCount){
        return ((C0+C1*key)% LARGE_PRIME )% bucketCount;
    }

    __forceinline__ __device__ unsigned int hash_g1(KEY_T key,unsigned int seed){
        return ((C10^seed+(C11^seed)*key)% LARGE_PRIME )%L2_SIZE;
    }
    __forceinline__ __device__ unsigned int hash_g2(KEY_T key,unsigned int seed){
        return ((C20^seed+(C21^seed)*key)% LARGE_PRIME )%L2_SIZE;
    }
    __forceinline__ __device__ unsigned int hash_g3(KEY_T key,unsigned int seed){
        return ((C30^seed+(C31^seed)*key)% LARGE_PRIME )%L2_SIZE;
    }

    __forceinline__ __device__ VALUE_T getHashValue(KEY_T key,KEY_PTR TK,VALUE_PTR TV,unsigned int *bucketSize, unsigned int bucketCount)
    {
        unsigned int bucket=hash_h(key,bucketCount);
        unsigned int l=0;
        unsigned int r=bucketSize[bucket];
        unsigned int mid;
        while(l<r)
        {
            mid =l+((r-l)/2);
            //if( (GET_HASH_KEY(T,bucket,mid)) < key){
            if( TK[GET_KEY_INDEX(bucket,mid)] <  key)
            {
                l=mid+1;
            }
            else
            {
                r=mid;
            }
        }
        //if(l < bucketSize[bucket] && GET_HASH_KEY(T,bucket,l)==key){
        if(l < bucketSize[bucket] && TK[GET_KEY_INDEX(bucket,l)]==key)
        {
        //	return GET_HASH_VALUE(T,bucket,l);
            return TV[GET_VALUE_INDEX(bucket,l)];
        }
        else
        {
            return MAX_INT;
        }
    }

    __global__ void debruijnCount(
        KEY_PTR lmerKeys,
        VALUE_PTR lmerValues,
        unsigned int lmerCount,
        KEY_PTR TK,
        VALUE_PTR TV,
        unsigned int * bucketSeed,
        unsigned int bucketCount,
        unsigned int * lcount,
        unsigned int * ecount,
        KEY_T validBitMask
    )
    {

        unsigned int tid = (blockDim.x * blockDim.y * gridDim.x * blockIdx.y)
                + (blockDim.x * blockDim.y * blockIdx.x)
                + (blockDim.x * threadIdx.y) + threadIdx.x;
        if (tid < lmerCount)
        {
            KEY_T lmer = lmerKeys[tid];
            VALUE_T lmerValue = lmerValues[tid];
            KEY_T prefix = (lmer & (validBitMask << 2)) >> 2;
            KEY_T suffix = (lmer & validBitMask);

            KEY_T lomask = 3;
            VALUE_T prefixIndex = getHashValue(prefix, TK, TV, bucketSeed,
                    bucketCount);
            VALUE_T suffixIndex = getHashValue(suffix, TK, TV, bucketSeed,
                    bucketCount);
            KEY_T transitionTo = (lmer & lomask);
            KEY_T transitionFrom = ((lmer >> __popcll(validBitMask)) & lomask);
            //atomicAdd(lcount+(prefixIndex<<2 )+transition,lmerValue);
            //atomicAdd(ecount+(suffixIndex<<2)+transition,lmerValue);
            lcount[(prefixIndex << 2) + transitionTo] = lmerValue;
            ecount[(suffixIndex << 2) + transitionFrom] = lmerValue;
        }
    }
    """, keep = True)
    d_lmerKeys = np.array(d_lmerKeys)
    d_lmerValues = np.array(d_lmerKeys)
    d_TK = np.array(d_TK)
    d_TV = np.array(d_TV)
    d_bucketSeed = np.array(d_bucketSeed)
    # d_lcount = np.array(d_lcount)
    # d_ecount = np.array(d_ecount)
    debruijn_count = mod.get_function("debruijnCount")
    block_dim, grid_dim = getOptimalLaunchConfiguration(lmerCount)
    logging.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    debruijn_count(
        drv.In(d_lmerKeys), drv.In(d_lmerValues), np.uint(lmerCount), drv.In(d_TK), drv.In(d_TV),
        drv.In(d_bucketSeed), np.uint(bucketCount), drv.Out(d_lcount), drv.Out(d_ecount),
        np.uint64(valid_bitmask), block = block_dim, grid = grid_dim
    )


def setup_vertices_device(d_kmerKeys, kmerCount, d_TK, d_TV, d_bucketSeed,
                          bucketCount, d_ev, d_lcount, d_lstart, d_ecount, d_estart):
    """
    This kernel works on a k-mer (l-1mer) which are vertices of the graph.
    :return:
    """

    logger = logging.getLogger('eulercuda.pydebruijn.setup_vertices_device')

    logger.info("started.")
    mod = SourceModule("""
    typedef unsigned long long  KEY_T ;
    typedef KEY_T               *KEY_PTR;
    typedef unsigned int        VALUE_T;
    typedef VALUE_T             *VALUE_PTR;

    typedef struct EulerVertex{
        KEY_T	vid;
        unsigned int  ep;
        unsigned int  ecount;
        unsigned int  lp;
        unsigned int  lcount;
    }EulerVertex;

    __global__ void setupVertices(KEY_PTR kmerKeys, unsigned int kmerCount,
            KEY_PTR TK, VALUE_PTR TV, unsigned int * bucketSeed,
            unsigned int bucketCount, EulerVertex * ev, unsigned int * lcount,
            unsigned int * loffset, unsigned int * ecount, unsigned int * eoffset)
    {
        unsigned int tid = (blockDim.x * blockDim.y * gridDim.x * blockIdx.y)
                + (blockDim.x * blockDim.y * blockIdx.x)
                + (blockDim.x * threadIdx.y) + threadIdx.x;
        if (tid < kmerCount)
        {
            KEY_T key = kmerKeys[tid];
            // from gpuhash_device.h
            VALUE_T index = getHashValue(key, TK, TV, bucketSeed, bucketCount);

            ev[index].vid = key;
            ev[index].lp = loffset[(index << 2)];
            ev[index].lcount = lcount[(index << 2)] + lcount[(index << 2) + 1]
                    + lcount[(index << 2) + 2] + lcount[(index << 2) + 3];
            ev[index].ep = eoffset[(index << 2)];
            ev[index].ecount = ecount[(index << 2)] + ecount[(index << 2) + 1]
                    + ecount[(index << 2) + 2] + ecount[(index << 2) + 3];
        }
    }
    """)
    setup_vertices = mod.get_function("setupVertices")
    block_dim, grid_dim = getOptimalLaunchConfiguration(kmerCount)
    logging.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    setup_vertices(drv.In(d_kmerKeys), np.uintt(kmerCount), drv.In(d_TK), drv.In(d_TK),
                   drv.In(d_bucketSeed), np.uint(bucketCount), drv.Out(d_ev), drv.In(d_lcount),
                   drv.In(d_lstart), drv.In(d_ecount),drv.In(d_estart), block = block_dim, grid = grid_dim
    )


def setup_edges_device(d_lmerKeys, d_lmerValues, d_lmerOffsets, lmerCount, d_TK, d_TV, d_bucketSeed, bucketCount,
                       d_l, d_e, d_ee, d_lstart, d_estart, validBitMask):
    """
    This kernel works on an l-mer, which represents an edge
    in the debruijn Graph.

    :return:
    """

    logger = logging.getLogger('eulercuda.pydebruijn.setup_edges_device')

    logger.info("started.")
    mod = SourceModule("""
    typedef unsigned long long  KEY_T ;
    typedef KEY_T               *KEY_PTR;
    typedef unsigned int        VALUE_T;
    typedef VALUE_T             *VALUE_PTR;

    typedef struct EulerEdge{
        KEY_T eid;
        unsigned int v1;
        unsigned int v2;
        unsigned int s;
        unsigned int pad;
    }EulerEdge;

    __global__ void setupEdges( KEY_PTR  lmerKeys,  VALUE_PTR  lmerValues,
             unsigned int *  lmerOffsets, const unsigned int lmerCount,
             KEY_PTR  TK, VALUE_PTR  TV, unsigned int *  bucketSeed,
            const unsigned int bucketCount, unsigned int *  l,
             unsigned int *  e, EulerEdge *  ee,
             unsigned int *  loffsets, unsigned int *  eoffsets,
            const KEY_T validBitMask)
    {

        unsigned int tid = (blockDim.x * blockDim.y * gridDim.x * blockIdx.y)
                + (blockDim.x * blockDim.y * blockIdx.x)
                + (blockDim.x * threadIdx.y) + threadIdx.x;
        if (tid < lmerCount)
        {
            KEY_T lmer = lmerKeys[tid];
            VALUE_T lmerValue = lmerValues[tid];
            KEY_T prefix = (lmer & (validBitMask << 2)) >> 2;
            KEY_T suffix = (lmer & validBitMask);
            KEY_T lomask = 3;
            //prefix and suffix index must be less than kmer count
            VALUE_T prefixIndex = getHashValue(prefix, TK, TV, bucketSeed,
                    bucketCount);
            VALUE_T suffixIndex = getHashValue(suffix, TK, TV, bucketSeed,
                    bucketCount);
            KEY_T transitionTo = (lmer & lomask);
            KEY_T transitionFrom = ((lmer >> __popcll(validBitMask)) & lomask);
            unsigned int loffset = loffsets[(prefixIndex << 2) + transitionTo];
            unsigned int eoffset = eoffsets[(suffixIndex << 2) + transitionFrom];

            unsigned int lmerOffset = lmerOffsets[tid];
            for (unsigned int i = 0; i < lmerValue; i++)
            {

                ee[lmerOffset].eid =lmerOffset;
                ee[lmerOffset].v1 = prefixIndex;
                ee[lmerOffset].v2 = suffixIndex;
                // lmerOffset;
                ee[lmerOffset].s = lmerValues[lmerCount - 1]
                        + lmerOffsets[lmerCount - 1];

                l[loffset] = lmerOffset;
                e[eoffset] = lmerOffset;
                loffset++;
                eoffset++;
                lmerOffset++;
            }
        }
    }
    """)
    setup_edges = mod.get_function("setupEdges")


    block_dim, grid_dim = getOptimalLaunchConfiguration(lmerCount)
    logging.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    setup_edges(
        drv.In(d_lmerKeys), drv.In(d_lmerValues), drv.InOut(d_lmerOffsets), np.uint(lmerCount), drv.In(d_TK),
        drv.In(d_TV), drv.In(d_bucketSeed), np.uint(bucketCount), drv.In(d_l), drv.In(d_e), drv.In(d_e),
        drv.Out(d_ee), drv.Out(d_lstart), drv.In(d_estart), drv.In(validBitMask), block = block_dim, grid = grid_dim
    )

def construct_debruijn_graph_device(d_lmerKeys, d_lmerValues, lmerCount, d_kmerKeys, kmerCount, l,
                                    d_TK, d_TV, d_bucketSeed, bucketCount, d_ev, d_l, d_e, d_ee, ecount):
    """

    :return:
    """
    logger = logging.getLogger('eulercuda.pydebruijn.construct_debruijn_graph_device')

    logger.info("started.")
    # pycuda has a parallel sum we can use instead of CUDPP. https://documen.tician.de/pycuda/array.html#module-pycuda.scan
    k = l - 1
    valid_bitmask = 0
    for i in range(k * 2):
        valid_bitmask = (valid_bitmask << 1) | 1
    logger.info("Bitmask = %s" % (valid_bitmask))

    # mem_size = (kmerCount) * sizeof(unsigned int) *4; // 4 - tuple for each kmer
    mem_size = kmerCount * sys.getsizeof(np.uint) * 4

    d_lcount = np.array(lmerCount, dtype = np.uint)
    d_lstart = np.empty_like(d_lcount)
    d_ecount = np.array(ecount, dtype = np.uint)
    d_estart = np.empty_like(d_ecount)

    # kernel call
    debruijn_count_device(d_lmerKeys, d_lmerValues, lmerCount, d_TK, d_TV, d_bucketSeed, bucketCount,
                          d_lcount, d_ecount, valid_bitmask)

    #  we need to perform pre-fix scan on , lcount, ecount, lmerValues,
    #  lcount and ecount has equal number of elements ,4*kmercount
    #  lmer has lmerCount elements, choose whichever is larger

    # configLmer.op = CUDPP_ADD;
    # configLmer.datatype = CUDPP_UINT;
    # configLmer.algorithm = CUDPP_SCAN;
    # configLmer.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;

    # cudppScan(scanplanKmer, d_lstart, d_lcount, 4 * kmerCount);
    # cudppScan(scanplanKmer, d_estart, d_ecount, 4 * kmerCount);

    knl = ExclusiveScanKernel(np.uint, "a+b")
    np_d_lcount = gpuarray(d_lcount)
    knl(np_d_lcount)
    np_d_lcount.get(d_lstart)

    np_d_ecount = gpuarray(d_ecount)
    knl(np_d_ecount)
    np_d_ecount.get(d_estart)

    d_lmerOffsets = np.empty(lmerCount * sys.getsizeof(np.uint64))
    np_d_lmerOffsets = knl(d_lmerValues)
    np_d_lmerOffsets.get(d_lmerOffsets)

    buffer = []
    buffer.append(d_lmerOffsets + lmerCount - 1)
    buffer.append(d_lmerValues + lmerCount - 1)
    ecount = buffer[0] + buffer[1]

    logger.info("debruijn vertex count:%d \ndebruijn edge count:%d" %
                (kmerCount, ecount))
# setupVertices<<<grid,block>>>(d_kmerKeys,kmerCount,d_TK,d_TV,d_bucketSeed,bucketCount,
    # *d_ev,d_lcount,d_lstart,d_ecount,d_estart);
    setup_vertices_device(d_kmerKeys, kmerCount, d_TK, d_TV, d_bucketSeed,
                          bucketCount, d_ev, d_lcount, d_lstart,d_ecount, d_estart)

    # setupEdges << < grid, block >> > (
    # d_lmerKeys, d_lmerValues, d_lmerOffsets, lmerCount, d_TK, d_TV, d_bucketSeed, bucketCount, *d_l, *d_e, *d_ee,
    # d_lstart, d_estart, validBitMask);
    setup_edges_device(d_lmerKeys,d_lmerValues,d_lmerOffsets,lmerCount, d_TK,d_TV,d_bucketSeed,
                       bucketCount,d_l,d_e,d_ee,d_lstart,d_estart,valid_bitmask)
    logger.info('Finished. Leaving.')