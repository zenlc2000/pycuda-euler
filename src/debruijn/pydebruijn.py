import sys
import numpy as np
import pycuda.driver as drv
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.scan import ExclusiveScanKernel
from pycuda.driver import device_attribute
import logging
from pyencode import getOptimalLaunchConfiguration

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

    __global__ void debruijnCount(
        KEY_PTR lmerKeys,                               /* lmer keys	*/
        VALUE_PTR lmerValues,                           /* lmer frequency */
        unsigned int lmerCount,                         /* total lmers */
        KEY_PTR TK,                                     /* Keys' pointer for Hash table*/
        VALUE_PTR TV,                                   /* Value pointer for Hash table*/
        unsigned int * bucketSeed,                      /* bucketSize: size of each bucket (it should be renamed to bucketSize)*/
        unsigned int bucketCount,                       /* total buckets */
        unsigned int * lcount,                          /* leaving edge count array : OUT */
        unsigned int * ecount,                          /* entering edge count array: OUT */
        KEY_T validBitMask                              /* bit mask for K length encoded bits*/
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
    """)

    debruijn_count = mod.get("debruijnCount")
    block_dim, grid_dim = getOptimalLaunchConfiguration(lmerCount)
    logging.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    debruijn_count(
        drv.In(d_lmerKeys), drv.In(d_lmerValues), np.uint(lmerCount), drv.In(d_TK), drv.In(d_TV),
        drv.In(d_bucketSeed), np.uint(bucketCount), drv.Out(d_lcount), drv.Out(d_ecount),
        np.uint64(valid_bitmask), block = block_dim, grid = grid_dim
    )


def setup_vertices_device():
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
    setup_vertices = mod.get("setupVertices")

def setup_edges_device():
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
    setup_edges = mod.get("setupEdges")


def construct_debruijn_graph_device(ecount, d_lmerKeys, d_lmerValues, lmerCount, d_kmerKeys, kmerCount,lmer_count, l,
                                    d_TK, d_TV, d_bucketSeed, bucketCount, d_ev, d_l, d_e, d_ee):
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

    d_lcount = np.empty(mem_size, dtype = np.uint)
    d_lstart = np.empty_like(d_lcount)
    d_ecount = np.empty(mem_size, dtype = np.uint)
    d_estart = np.empty_like(d_ecount)
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