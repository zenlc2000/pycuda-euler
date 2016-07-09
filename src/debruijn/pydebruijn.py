import sys
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.scan import ExclusiveScanKernel
import logging


module_logger = logging.getLogger('eulercuda.pydebruijn')


def debruijn_count_device(d_lmerKeys, d_lmerValues, lmerCount, d_TK, d_TV, d_bucketSize, bucketCount,
                          d_lcount, d_ecount, valid_bitmask, readLength, readCount):
    """

    This kernel works on each l-mer ,counting edges of the graph.


    :return:
    """
    logger = logging.getLogger('eulercuda.pydebruijn.debruijn_count')
    logger.info("started.")
    mod = SourceModule("""
    #//include <stdio.h>
    typedef unsigned long long  KEY_T ;
    typedef KEY_T               *KEY_PTR;
    typedef unsigned int       VALUE_T;
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


    __forceinline__ __device__ unsigned int hash_h(KEY_T  key, unsigned int bucketCount)
    {
        return ((C0 + C1 * key) % LARGE_PRIME) % bucketCount;
    }
/*
    __forceinline__ __device__ unsigned int hash_g1(KEY_T key,unsigned int seed)
    {
        return ((C10^seed + (C11^seed) * key) % LARGE_PRIME) % L2_SIZE;
    }
    __forceinline__ __device__ unsigned int hash_g2(KEY_T key,unsigned int seed)
    {
        return ((C20^seed + (C21^seed) * key) % LARGE_PRIME) % L2_SIZE;
    }
    __forceinline__ __device__ unsigned int hash_g3(KEY_T key,unsigned int seed)
    {
        return ((C30^seed + (C31^seed) * key) % LARGE_PRIME) % L2_SIZE;
    }
*/
    __forceinline__ __device__
    VALUE_T getHashValue(KEY_T key, KEY_PTR TK, VALUE_PTR TV, unsigned int *bucketSize,
                                        unsigned int bucketCount)
    {
       // printf(" key = %llu, bucketCount = %d ", key, bucketCount);
        unsigned int bucket = hash_h(key,bucketCount);
        unsigned int l = 0;
        unsigned int r = bucketSize[bucket];
        unsigned int mid;
        while(l < r)
        {
            mid = l + ((r - l) / 2);
            if( TK[GET_KEY_INDEX(bucket, mid)] <  key)
            {
                l = mid + 1;
            }
            else
            {
                r = mid;
            }
        }
        if(l < bucketSize[bucket] && TK[GET_KEY_INDEX(bucket, l)] == key)
        {
            //printf(" TV[GET_VALUE_INDEX(bucket, l)] = %u ", TV[GET_VALUE_INDEX(bucket, l)]);
            return TV[GET_VALUE_INDEX(bucket, l)];
        }
        else
        {
           // printf(" key = %llu MAX_INT = %u ", key, MAX_INT);
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
                    KEY_T validBitMask)
    {

        unsigned int tid = (blockDim.x * blockDim.y * gridDim.x * blockIdx.y)
                        + (blockDim.x * blockDim.y * blockIdx.x)
                        + (blockDim.x * threadIdx.y) + threadIdx.x;

        if (tid < lmerCount)
        {
            KEY_T lmer = lmerKeys[tid];

            VALUE_T lmerValue = lmerValues[tid];
           // printf(" lmerValue = %u ", lmerValue);

            KEY_T prefix = (lmer & (validBitMask << 2)) >> 2;
            KEY_T suffix = (lmer & validBitMask);
            //printf(" prefix= %llu, suffix = %llu ", prefix, suffix);
            KEY_T lomask = 3;
            VALUE_T prefixIndex = getHashValue(prefix, TK, TV, bucketSeed, bucketCount);
            VALUE_T suffixIndex = getHashValue(suffix, TK, TV, bucketSeed, bucketCount);
            //printf(" prefixIndex = %u, suffixIndex = %u ", prefixIndex, suffixIndex);
            //printf(" prefixIndex << 2 = %u, suffixIndex << 2 = %u ", (prefixIndex << 2), (suffixIndex << 2));

            KEY_T transitionTo = (lmer & lomask);
            KEY_T transitionFrom = ((lmer >> __popcll(validBitMask)) & lomask);

            //printf(" transitionTo = %llu, transitionFrom = %llu ",transitionTo, transitionFrom);

            lcount[(prefixIndex << 2) + transitionTo] = lmerValue;
            ecount[(suffixIndex << 2) + transitionFrom] = lmerValue;

            //printf(" lcountIndex = %llu, ecountIndex = %llu ", (prefixIndex << 2)+ transitionTo, (suffixIndex << 2)+ transitionFrom);
            //printf(" lcount = %u, ecount = %u ",  lcount[(prefixIndex << 2) + transitionTo], ecount[(suffixIndex << 2) + transitionFrom]);
        }
    }
    """, keep=True)
    d_lmerKeys = np.array(d_lmerKeys, dtype=np.ulonglong)
    d_lmerValues = np.array(d_lmerValues, dtype=np.uintc)

    np_d_lcount = gpuarray.to_gpu(d_lcount)
    np_d_ecount = gpuarray.to_gpu(d_ecount)
    debruijn_count = mod.get_function("debruijnCount")
    block_dim, grid_dim = getOptimalLaunchConfiguration(lmerCount,512)
    logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    debruijn_count(
        drv.In(d_lmerKeys),
        drv.In(d_lmerValues),
        np.uintc(lmerCount),
        drv.In(d_TK),
        drv.In(d_TV),
        drv.In(d_bucketSize),
        np.uintc(bucketCount),
        np_d_lcount,
        np_d_ecount,
        np.ulonglong(valid_bitmask),
        block=block_dim, grid=grid_dim
    )
    np_d_lcount.get(d_lcount)
    np_d_ecount.get(d_ecount)
    devdata = pycuda.tools.DeviceData()
    orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[1])
    logger.info("Occupancy = %s" % (orec.occupancy * 100))

    logger.info("Finished. Leaving.")
    return d_lcount, d_ecount


def setup_vertices_device(d_kmerKeys, kmerCount, d_TK, d_TV, d_bucketSeed,
                          bucketCount, d_ev, d_lcount, d_lstart, d_ecount, d_estart):
    """
    This kernel works on a k-mer (l-1mer) which are vertices of the graph.
    :return:
    """

    logger = logging.getLogger('eulercuda.pydebruijn.setup_vertices_device')

    logger.info("started.")
    mod = SourceModule("""
    //#include <stdio.h>
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


    __forceinline__ __device__ unsigned int hash_h(KEY_T  key, unsigned int bucketCount)
    {
        return ((C0 + C1 * key) % LARGE_PRIME) % bucketCount;
    }

    __forceinline__ __device__ VALUE_T getHashValue(KEY_T key, KEY_PTR TK, VALUE_PTR TV, unsigned int *bucketSize,
                                        unsigned int bucketCount)
    {
       // printf(" key = %llu, bucketCount = %d ", key, bucketCount);
        unsigned int bucket = hash_h(key,bucketCount);
        unsigned int l = 0;
        unsigned int r = bucketSize[bucket];
        unsigned int mid;
        while(l < r)
        {
            mid = l + ((r - l) / 2);
            if( TK[GET_KEY_INDEX(bucket, mid)] <  key)
            {
                l = mid + 1;
            }
            else
            {
                r = mid;
            }
        }
        if(l < bucketSize[bucket] && TK[GET_KEY_INDEX(bucket, l)] == key)
        {
          //  printf(" TV[GET_VALUE_INDEX(bucket, l)] = %d ", TV[GET_VALUE_INDEX(bucket, l)]);
            return TV[GET_VALUE_INDEX(bucket, l)];
        }
        else
        {
            //printf(" MAX_INT = %u ", MAX_INT);
            return MAX_INT;
        }
    }

/*__global__ void setupVertices(
                                   KEY_PTR kmerKeys,
                                    unsigned int kmerCount,
                                    KEY_PTR TK,
                                    VALUE_PTR TV,
                                    unsigned int * bucketSeed,
                                    unsigned int bucketCount,
                                    KEY_PTR ev_vid,
                                    unsigned int * ev_ep,
                                    unsigned int * ev_ecount,
                                    unsigned int * ev_lp,
                                    unsigned int * ev_lcount,
                                    unsigned int * lcount,
                                    unsigned int * loffset,
                                    unsigned int * ecount,
                                    unsigned int * eoffset)*/

    __global__ void setupVertices(KEY_PTR kmerKeys, unsigned int kmerCount,
            KEY_PTR TK, VALUE_PTR TV, unsigned int * bucketSeed,
            unsigned int bucketCount, EulerVertex * ev, unsigned int * lcount,
            unsigned int * loffset, unsigned int * ecount, unsigned int * eoffset)

    {
        unsigned int tid = (blockDim.x * blockDim.y * gridDim.x * blockIdx.y)
                + (blockDim.x * blockDim.y * blockIdx.x)
                + (blockDim.x * threadIdx.y) + threadIdx.x;
        if (tid < kmerCount) {
            KEY_T key = kmerKeys[tid];
            VALUE_T index = getHashValue(key, TK, TV, bucketSeed, bucketCount);
            ;
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
    block_dim, grid_dim = getOptimalLaunchConfiguration(kmerCount, 512)
    logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    setup_vertices(
        drv.In(d_kmerKeys),
        np.uintc(kmerCount),
        drv.In(d_TK),
        drv.In(d_TV),
        drv.In(d_bucketSeed),
        np.uintc(bucketCount),
        drv.Out(d_ev),
        drv.In(d_lcount),
        drv.In(d_lstart),
        drv.In(d_ecount),
        drv.In(d_estart),
        block=block_dim, grid=grid_dim
                    )
    # d_ev.copy_from_gpu()
    devdata = pycuda.tools.DeviceData()
    orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[1])
    logger.info("Occupancy = %s" % (orec.occupancy * 100))
    logger.info("Finished. Leaving.")


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
    //#include <stdio.h>
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
    #define LARGE_PRIME 1900813
    #define L2_SIZE 192
    #define MAX_ITERATIONS 100
    #define MAX_INT 0xffffffff
    #define MAX_SEED_COUNT 25
    #define C0  0x01010101
    #define C1	0x12345678

    #define GET_KEY_INDEX(blockIdx,itemIdx) ((blockIdx)*MAX_BUCKET_ITEM+(itemIdx))
    #define GET_VALUE_INDEX(blockIdx,itemIdx) ((blockIdx)*MAX_BUCKET_ITEM+(itemIdx))
    #define MAX_BUCKET_ITEM (520)


    __forceinline__ __device__ unsigned int hash_h(KEY_T  key, unsigned int bucketCount)
    {
        return ((C0 + C1 * key) % LARGE_PRIME) % bucketCount;
    }

    __forceinline__ __device__
    VALUE_T getHashValue(KEY_T key, KEY_PTR TK, VALUE_PTR TV, unsigned int *bucketSize,
                                        unsigned int bucketCount)
    {
       // printf(" key = %llu, bucketCount = %d ", key, bucketCount);
        unsigned int bucket = hash_h(key,bucketCount);
        //printf(" bucket = %u ", bucket);
        unsigned int l = 0;
        unsigned int r = bucketSize[bucket];
        unsigned int mid;
        while(l < r)
        {
            mid = l + ((r - l) / 2);
            if( TK[GET_KEY_INDEX(bucket, mid)] <  key)
            {
                l = mid + 1;
            }
            else
            {
                r = mid;
            }
        }
        if(l < bucketSize[bucket] && TK[GET_KEY_INDEX(bucket, l)] == key)
        {
          //  printf(" TV[GET_VALUE_INDEX(bucket, l)] = %d ", TV[GET_VALUE_INDEX(bucket, l)]);
            return TV[GET_VALUE_INDEX(bucket, l)];
        }
        else
        {
            //printf(" MAX_INT = %u ", MAX_INT);
            return MAX_INT;
        }
    }

 /*   __global__ void setupEdges(
                KEY_PTR  lmerKeys,
                VALUE_PTR  lmerValues,
                unsigned int *  lmerOffsets,
                const unsigned int lmerCount,
                KEY_PTR  TK,
                VALUE_PTR  TV,
                unsigned int *  bucketSeed,
                const unsigned int bucketCount,
                unsigned int *  l,
                unsigned int *  e,
                KEY_PTR ee_eid,
                unsigned int * ee_v1,
                unsigned int * ee_v2,
                unsigned int * ee_s,
                unsigned int * ee_pad,
                unsigned int *  loffsets,
                unsigned int *  eoffsets,
                const KEY_T validBitMask) */

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
        if (tid < lmerCount) {
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
            for (unsigned int i = 0; i < lmerValue; i++) {

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
    }    """)
    setup_edges = mod.get_function("setupEdges")

    block_dim, grid_dim = getOptimalLaunchConfiguration(lmerCount, 512)
    logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    setup_edges(
        drv.In(d_lmerKeys),
        drv.In(d_lmerValues),
        drv.InOut(d_lmerOffsets),
        np.uintc(lmerCount),
        drv.In(d_TK),
        drv.In(d_TV),
        drv.In(d_bucketSeed),
        np.uintc(bucketCount),
        drv.Out(d_l),
        drv.Out(d_e),
        drv.Out(d_ee),
        drv.In(d_lstart),
        drv.In(d_estart),
        np.ulonglong(validBitMask),
        block=block_dim, grid=grid_dim
    )
    devdata = pycuda.tools.DeviceData()
    orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[1])
    logger.info("Occupancy = %s" % (orec.occupancy * 100))
    logger.info("Finished. Leaving.")


def construct_debruijn_graph_device(d_lmerKeys, d_lmerValues, lmerCount, d_kmerKeys, kmerCount, l,
                                    d_TK, d_TV, d_bucketSeed, bucketCount, d_ev, d_l, d_e, d_ee,
                                    readLength, readCount):
    """

    :return:
    """
    logger = logging.getLogger('eulercuda.pydebruijn.construct_debruijn_graph_device')

    logger.info("started.")
    # pycuda has a parallel sum we can use instead of CUDPP.
    # https://documen.tician.de/pycuda/array.html#module-pycuda.scan
    k = l - 1
    valid_bitmask = 0
    for i in range(k * 2):
        valid_bitmask = (valid_bitmask << 1) | 1
    logger.info("Bitmask = %s" % valid_bitmask)

    # mem_size = (kmerCount) * sizeof(unsigned int) *4; // 4 - tuple for each kmer
    mem_size = kmerCount * sys.getsizeof(np.uintc) * 4

    d_lcount = np.zeros(mem_size, dtype=np.uintc)

    d_ecount = np.zeros_like(d_lcount)
    d_lstart = np.zeros_like(d_lcount)
    d_estart = np.zeros_like(d_lcount)

    # kernel call
    d_lcount, d_ecount = debruijn_count_device(d_lmerKeys, d_lmerValues, lmerCount, d_TK, d_TV, d_bucketSeed, bucketCount,
                          d_lcount, d_ecount, valid_bitmask, readLength, readCount)

    #  we need to perform pre-fix scan on , lcount, ecount, lmerValues,
    #  lcount and ecount has equal number of elements ,4*kmercount
    #  lmer has lmerCount elements, choose whichever is larger


    # d_lcount = [l for l in d_lcount.tolist() if l < 1000]
    # d_ecount = [e for e in d_ecount.tolist() if e < 1000]
    knl = ExclusiveScanKernel(np.uintc, "a+b", 0)
    np_d_lcount = gpuarray.to_gpu(np.array(d_lcount, dtype=np.uintc))
    knl(np_d_lcount)
    np_d_lcount.get(d_lstart)

    np_d_ecount = gpuarray.to_gpu(np.array(d_ecount, dtype=np.uintc))
    knl(np_d_ecount)
    np_d_ecount.get(d_estart)

    np_d_lmerValues = np.array(d_lmerValues, dtype=np.uintc)
    d_lmerOffsets = np.zeros_like(np_d_lmerValues) #(sys.getsizeof(np.uintc) * 4 * kmerCount, dtype=np.uintc)
    np_d_lmerValues = gpuarray.to_gpu(np_d_lmerValues)
    knl(np_d_lmerValues)
    np_d_lmerValues.get(d_lmerOffsets)
    ecount = []
    [ecount.append(l) for l in d_lmerOffsets]
    [ecount.append(v) for v in d_lmerValues]
    logger.info("debruijn vertex count:%s \ndebruijn edge count:%s" % (kmerCount, len(ecount)))

    d_kmerKeys = np.array(d_kmerKeys, dtype=np.ulonglong)

    struct_size = sys.getsizeof(np.ulonglong) + (4 * sys.getsizeof(np.uintc))
    # d_ev = {
    #     'vid': np.zeros(struct_size, dtype=np.ulonglong),
    #     'ep': np.zeros(struct_size, dtype=np.uintc),
    #     'ecount': np.zeros(struct_size, dtype=np.uintc),
    #     'lp': np.zeros(struct_size, dtype=np.uintc),
    #     'lcount': np.zeros(struct_size, dtype=np.uintc)
    # }
    # d_ee = {
    #     'eid': np.zeros(struct_size, dtype=np.ulonglong),
    #     'v1': np.zeros(struct_size, dtype=np.uintc),
    #     'v2': np.zeros(struct_size, dtype=np.uintc),
    #     's': np.zeros(struct_size, dtype=np.uintc),
    #     'pad': np.zeros(struct_size, dtype=np.uintc)
    # }
    d_ee = np.zeros(struct_size, dtype=[('eid',np.ulonglong), ('v1',np.uintc), ('v2',np.uintc), ('s',np.uintc), ('pad',np.uintc)])
    d_ev = np.zeros(struct_size, dtype=[('vid',np.ulonglong), ('ep',np.uintc), ('ecount',np.uintc), ('lp',np.uintc), ('lcount',np.uintc)])
    setup_vertices_device(d_kmerKeys, kmerCount, d_TK, d_TV, d_bucketSeed,
                          bucketCount, d_ev, d_lcount, d_lstart, d_ecount, d_estart)

    d_l = np.zeros(sys.getsizeof(np.uintc) * len(ecount), dtype=np.uintc)
    d_e = np.zeros(sys.getsizeof(np.uintc) * len(ecount), dtype=np.uintc)
    d_lmerKeys = np.array(d_lmerKeys, dtype=np.ulonglong)
    d_lmerValues = np.array(d_lmerValues, dtype=np.uintc)
    setup_edges_device(d_lmerKeys, d_lmerValues, d_lmerOffsets, lmerCount, d_TK, d_TV, d_bucketSeed,
                       bucketCount, d_l, d_e, d_ee, d_lstart, d_estart, valid_bitmask)
    logger.info('Finished. Leaving.')
    return d_ee, d_ev, d_l, d_e, kmerCount, len(ecount)


def getOptimalLaunchConfiguration(threadCount, threadPerBlock=32):
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
