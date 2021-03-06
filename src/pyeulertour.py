""" GPU Accelerated Euler Tour """
import sys
import numpy as np
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.scan import ExclusiveScanKernel
import logging
from encoder.pyencode import getOptimalLaunchConfiguration
from component.pycomponent import find_component_device

module_logger = logging.getLogger('eulercuda.pyeulertour')
# ULONGLONG = 8
# UINTC = 4

def assign_successor_device(d_ev, d_l, d_e, vcount, d_ee, ecount):
    """

    :param d_ev:
    :param d_l:
    :param d_e:
    :param vcount:
    :param d_ee:
    :param ecount:
    :return:
    """
    # logger = logging.getLogger('eulercuda.pyeulertour.assign_successor_device')
    module_logger.info("started assign_successor_device.")
    mod = SourceModule("""
    #include <stdio.h>
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

    typedef struct EulerVertex{
        KEY_T	vid;
        unsigned int  ep;
        unsigned int  ecount;
        unsigned int  lp;
        unsigned int  lcount;
    }EulerVertex;

    __global__  void assignSuccessor(
                        EulerVertex * ev,
                        unsigned int * l,
                        unsigned int * e,
                        unsigned vcount,
                        EulerEdge * ee ,
                        unsigned int ecount)
    {
        unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
        unsigned int eidx = 0;
        if(tid < vcount)
        {
            while(eidx < ev[tid].ecount && eidx < ev[tid].lcount)
            {
                unsigned int eindex, lindex, eeindex;
                eindex = ev[tid].ep + eidx;
                lindex = ev[tid].lp + eidx;
                if (eindex < ecount)
                {

                    eeindex = e[ev[tid].ep + eidx];
                    if (eindex < ecount && lindex < ecount && eeindex < ecount)
                    {
                      //  printf(" e = %u, l = %u, ee = %u ", eindex, lindex, eeindex);
                        ee[e[ev[tid].ep + eidx]].s = l[ev[tid].lp + eidx] ;
                    }
                }
                eidx++;
            }
        }
    }
    """)
    free, total = drv.mem_get_info()
    # module_logger.debug(" %s free out of %s total memory" % (free, total) )
    block_dim, grid_dim = getOptimalLaunchConfiguration(vcount, 256)
    module_logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    np_d_ev = gpuarray.to_gpu(d_ev)
    np_d_ee = gpuarray.to_gpu(d_ee)
    assign_successor = mod.get_function("assignSuccessor")
    assign_successor(             # ecount is list - should be uint
        np_d_ev,
        drv.In(d_l),
        drv.In(d_e),
        np.uintc(vcount),
        np_d_ee,
        np.uintc(ecount),
        block=block_dim, grid=grid_dim
    )
    # devdata = pycuda.tools.DeviceData()
    # orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[1])
    # module_logger.info("Occupancy = %s" % (orec.occupancy * 100))
    np_d_ev.get(d_ev)
    np_d_ee.get(d_ee)
    module_logger.info("Finished. Leaving.")
    return d_ev, d_ee

def construct_successor_graphP1_device(d_ee, d_v, ecount):
    logger = logging.getLogger('eulercuda.pyeulertour.construct_successor_graphP1_device')
    logger.info("started.")
    mod = SourceModule("""
    #include <stdio.h>
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
    typedef struct Vertex{
        unsigned int vid;
        unsigned int n1;
        unsigned int n2;
    } Vertex;


    __global__ void constructSuccessorGraphP1(EulerEdge* e, Vertex * v, unsigned int ecount)
    {
        unsigned int tid = (blockDim.x*blockDim.y * gridDim.x*blockIdx.y) +
        (blockDim.x*blockDim.y*blockIdx.x) + (blockDim.x*threadIdx.y) + threadIdx.x;

        if(tid < ecount){
            v[tid].n1 = ecount;
            v[tid].n2 = ecount;
            v[tid].vid = e[tid].eid;
            v[tid].n1 = e[tid].s;
        }
    }
    """)

    construct_successor_graphP1 = mod.get_function("constructSuccessorGraphP1")
    block_dim, grid_dim = getOptimalLaunchConfiguration(ecount, 512)
    np_d_v = gpuarray.to_gpu(d_v)
    construct_successor_graphP1(
        drv.In(d_ee),
        np_d_v,
        np.uintc(ecount),
        block=block_dim, grid=grid_dim
    )
    # devdata = pycuda.tools.DeviceData()
    # orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[1])
    # logger.info("Occupancy = %s" % (orec.occupancy * 100))
    np_d_v.get(d_v)
    logger.info("Finished. Leaving.")
    return d_v


def construct_successor_graphP2_device(d_ee, d_v, ecount):
    logger = logging.getLogger('eulercuda.pyeulertour.construct_successor_graphP1_device')
    logger.info("started.")
    mod = SourceModule("""
    #include <stdio.h>
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
    typedef struct Vertex{
        unsigned int vid;
        unsigned int n1;
        unsigned int n2;
    } Vertex;

    __global__ void constructSuccessorGraphP2(EulerEdge* e, Vertex * v, unsigned int ecount)
    {
        unsigned int tid = (blockDim.x * blockDim.y * gridDim.x * blockIdx.y) + (blockDim.x*blockDim.y * blockIdx.x) +
        (blockDim.x * threadIdx.y) + threadIdx.x;

        if(tid<ecount)
        {
            if(v[tid].n1 <ecount )
            {
                v[v[tid].n1].n2=v[tid].vid;
            }
        }
    }
    """)
    construct_successor_graphP2 = mod.get_function("constructSuccessorGraphP2")
    block_dim, grid_dim = getOptimalLaunchConfiguration(ecount, 512)
    logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    np_d_v = gpuarray.to_gpu(d_v)
    construct_successor_graphP2(
        drv.In(d_ee),
        np_d_v,
        np.uintc(ecount),
        block=block_dim, grid=grid_dim
    )
    # devdata = pycuda.tools.DeviceData()
    # orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[1])
    # logger.info("Occupancy = %s" % (orec.occupancy * 100))
    np_d_v.get(d_v)
    logger.info("Finished. Leaving.")
    return d_v


def calculate_circuit_graph_vertex_data_device(d_D, d_C, length):
    logger = logging.getLogger('eulercuda.pyeulertour.calculate_circuit_graph_vertex_data_device')
    logger.info("started.")
    mod = SourceModule("""
    __global__ void calculateCircuitGraphVertexData( unsigned int * D,unsigned int * C,unsigned int ecount){

        unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
        if( tid <ecount)
        {
            unsigned int c=D[tid];
            atomicExch(C+c,1);
        }
    }
    """)
    calculate_circuit_graph_vertex_data = mod.get_function('calculateCircuitGraphVertexData')
    block_dim, grid_dim = getOptimalLaunchConfiguration(length, 512)
    logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    np_d_D = gpuarray.to_gpu(d_D)
    np_d_C = gpuarray.to_gpu(d_C)
    calculate_circuit_graph_vertex_data(
        np_d_D,
        np_d_C,
        np.uintc(length),
        block=block_dim, grid=grid_dim
    )
    np_d_D.get(d_D)
    np_d_C.get(d_C)
    # devdata = pycuda.tools.DeviceData()
    # orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[1])
    # logger.info("Occupancy = %s" % (orec.occupancy * 100))
    logger.info("Finished. Leaving.")
    return d_D, d_C

# typedef struct Vertex{
# 	unsigned int vid;
# 	unsigned int n1;
# 	unsigned int n2;
#
# } Vertex;

#    typedef struct EulerEdge{
#         KEY_T eid;
#         unsigned int v1;
#         unsigned int v2;
#         unsigned int s;
#         unsigned int pad;
#     }EulerEdge;


def construct_circuit_Graph_vertex(d_C, d_cg_offset, ecount, d_cv):
    """

    :param d_C:
    :param d_cg_offset:
    :param ecount:
    :param d_cv:
    :return:
    """
    logger = logging.getLogger('eulercuda.pyeulertour.construct_circuit_Graph_vertex')
    logger.info("started.")
    mod = SourceModule("""
        __global__ void constructCircuitGraphVertex(unsigned int * C,unsigned int * offset,unsigned int ecount, unsigned int * cv)
        {
            unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
            if(tid < ecount){
                if(C[tid] != 0){
                    cv[offset[tid]] = tid;
                }
            }
        }
    """)
    np_d_cv = gpuarray.to_gpu(d_cv)
    circuit_graph_vertex = mod.get_function('constructCircuitGraphVertex')
    block_dim, grid_dim = getOptimalLaunchConfiguration(ecount, 512)
    circuit_graph_vertex(
        drv.In(d_C),
        drv.In(d_cg_offset),
        np.uintc(ecount),
        np_d_cv,
        block=block_dim, grid=grid_dim
    )
    np_d_cv.get(d_cv)
    # devdata = pycuda.tools.DeviceData()
    # orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[1])
    # logger.info("Occupancy = %s" % (orec.occupancy * 100))
    return d_cv


def calculate_circuit_graph_edge_data(d_ev, d_e, vcount, d_D, d_cg_offset, ecount, d_cedgeCount ):
    """

    :param d_ev:
    :param d_e:
    :param vcount:
    :param d_D:
    :param d_cg_offset:
    :param ecount:
    :param d_cedgeCount:
    :return:
    """
    logger = logging.getLogger('eulercuda.pyeulertour.calculate_circuit_graph_edge_data')
    logger.info("started.")
    mod = SourceModule("""
    #include <stdio.h>
        typedef unsigned long long  KEY_T;
        typedef struct EulerVertex{
            KEY_T	vid;
            unsigned int  ep;
            unsigned int  ecount;
            unsigned int  lp;
            unsigned int  lcount;
        }EulerVertex;
        __global__ void calculateCircuitGraphEdgeData(
                                    EulerVertex* v,
                                    unsigned int * e,
                                    unsigned vCount,
                                    unsigned int * D,
                                    unsigned int * map,
                                    unsigned int ecount,
                                    unsigned int * cedgeCount)
        {

            unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
            unsigned int index = 0;
            unsigned int maxIndex = 0;
            unsigned int c1;
            unsigned int c2;
            index = 0;
            maxIndex = 0;
            if(tid < vCount && v[tid].ecount > 0 )
            {
                index = v[tid].ep;
                maxIndex = index + v[tid].ecount - 1;
               // printf(" index = %u, max = %u ", index, maxIndex);
                while (index < maxIndex && index < ecount )
                {

                    if (e[index] < ecount && e[index + 1] < ecount)
                    {
                       // printf(" map = %u, D = %u ",  map[D[e[index]]], D[e[index]]);
                        c1 = map[D[e[index]]];
                        c2 = map[D[e[index + 1]]];
                        if( c1 != c2)
                        {
                            unsigned int c = min(c1, c2);
                            atomicInc(cedgeCount + c, ecount);
                        }
                    }
                    index++;
                }
            }

        }
    """)
    circuit_graph_edge = mod.get_function('calculateCircuitGraphEdgeData')
    np_d_cedgeCount = gpuarray.to_gpu(d_cedgeCount)
    block_dim, grid_dim = getOptimalLaunchConfiguration(vcount, 512)
    circuit_graph_edge(
        drv.In(d_ev),
        drv.In(d_e),
        np.uintc(vcount),
        drv.In(d_D),
        drv.In(d_cg_offset),
        np.uintc(ecount),
        np_d_cedgeCount,
        block=block_dim, grid=grid_dim
    )
    np_d_cedgeCount.get(d_cedgeCount)
    # devdata = pycuda.tools.DeviceData()
    # orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[1])
    # logger.info("Occupancy = %s" % (orec.occupancy * 100))
    return d_cedgeCount


def assign_circuit_graph_edge_data(d_ev, d_e, vcount, d_D, d_cg_offset, ecount, d_cg_edge_start, d_cedgeCount,
                               circuitVertexSize, d_cg_edge, circuitGraphEdgeCount):
    """
    :param d_ev:
    :param d_e:
    :param vcount:
    :param d_D:
    :param d_cg_offset:
    :param ecount:
    :param d_cg_edge_start:
    :param d_cedgeCount:
    :param circuitVertexSize:
    :param d_cg_edge:
    :param circuitGraphEdgeCount:
    :return:
    """
    logger = logging.getLogger('eulercuda.pyeulertour.assign_circuit_graph_edge_data')
    logger.info("started.")
    mod = SourceModule("""
    typedef unsigned long long  KEY_T ;
    typedef struct EulerVertex{
        KEY_T	vid;
        unsigned int  ep;
        unsigned int  ecount;
        unsigned int  lp;
        unsigned int  lcount;
    }EulerVertex;
    typedef struct CircuitEdge{
        unsigned int ceid;
        unsigned e1;
        unsigned e2;
        unsigned c1;
        unsigned c2;
    }CircuitEdge;

    __global__ void assignCircuitGraphEdgeData(EulerVertex* v,
                           unsigned int * e,
                           unsigned vCount,
                           unsigned int * D,
                           unsigned int * map,
                           unsigned int ecount,
                           unsigned int * cedgeOffset,
                           unsigned int * cedgeCount,
                           unsigned int cvCount,
                           CircuitEdge * cedge,
                           unsigned int cecount)
    {

        unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+
                        (blockDim.x*threadIdx.y)+threadIdx.x;
        unsigned int index=0;
        unsigned int maxIndex=0;
        if(tid<vCount && v[tid].ecount>0){
            index=v[tid].ep;
            maxIndex=index+v[tid].ecount-1;
            while (index<maxIndex  && index < ecount )
            {
                if (e[index] < ecount && e[index + 1] < ecount)
                {
                    unsigned int c1=map[D[e[index]]];
                    unsigned int c2=map[D[e[index+1]]];
                    if( c1 != c2)
                    {
                        unsigned int c=min(c1,c2);
                        unsigned int t=max(c1,c2);
                        unsigned int i=atomicDec(cedgeCount+c,ecount);
                        i=i-1;
                        cedge[cedgeOffset[c]+i].c1=c;
                        cedge[cedgeOffset[c]+i].c2=t;
                        cedge[cedgeOffset[c]+i].e1=e[index];
                        cedge[cedgeOffset[c]+i].e2=e[index+1];
                    }
                }
                index++;
            }
        }
    }
    """)
    block_dim, grid_dim = getOptimalLaunchConfiguration(vcount, 512)
    np_d_cg_edge = gpuarray.to_gpu(d_cg_edge)
    cged = mod.get_function('assignCircuitGraphEdgeData')
    cged(
        drv.In(d_ev),
        drv.In(d_e),
        np.uintc(vcount),
        drv.In(d_D),
        drv.In(d_cg_offset),
        np.uintc(ecount),
        drv.In(d_cg_edge_start),
        drv.In(d_cedgeCount),
        np.uintc(circuitVertexSize),
        np_d_cg_edge,
        np.uintc(circuitGraphEdgeCount),
        block=block_dim, grid=grid_dim
    )
    np_d_cg_edge.get(d_cg_edge)
    # devdata = pycuda.tools.DeviceData()
    # orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[1])
    # logger.debug("Occupancy = %s" % (orec.occupancy * 100))
    logger.info('Finished.')
    return d_cg_edge

def execute_swipe(d_ev, d_e, vcount, d_ee, d_mark, ecount):
    logger = logging.getLogger('eulercuda.pyeulertour.execute_swipe')
    logger.info("started.")
    mod = SourceModule("""
    typedef unsigned long long  KEY_T ;
    typedef struct EulerVertex
    {
        KEY_T	vid;
        unsigned int  ep;
        unsigned int  ecount;
        unsigned int  lp;
        unsigned int  lcount;
    } EulerVertex;

    typedef struct EulerEdge
    {
        KEY_T eid;
        unsigned int v1;
        unsigned int v2;
        unsigned int s;
        unsigned int pad;
    }EulerEdge;

    __global__ void executeSwipe(
                                EulerVertex * ev,
                                unsigned int * e,
                                unsigned int vcount ,
                                EulerEdge * ee,
                                unsigned int * mark,
                                unsigned int ecount)
        {

        unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
        unsigned int t;
        unsigned int index=0;
        unsigned int maxIndex;
        unsigned int s;
        if (tid < vcount)
        {
            index = ev[tid].ep;
            maxIndex = index + ev[tid].ecount - 1;
            while (index < maxIndex && ee[e[index]].eid < ecount)
            {

              /*  if (mark[ee[e[index]].eid] == 1)
                {
                    t = index;
                    s = ee[e[index]].s;
                    while (mark[ee[e[index]].eid] == 1 && index < maxIndex)
                    {
                        ee[e[index]].s = ee[e[index+1]].s;
                        index = index + 1;
                    }
                    if(t != index)
                    {
                        ee[e[index]].s = s;
                    }
                }  */
                index++;
            }

        }
    }



    """)
    block_dim, grid_dim = getOptimalLaunchConfiguration(vcount.item(), 512)
    np_d_mark = gpuarray.to_gpu(d_mark)
    np_d_ee = gpuarray.to_gpu(d_ee)
    swipe = mod.get_function('executeSwipe')
    swipe(
        drv.In(d_ev),
        drv.In(d_e),
        np.uintc(vcount),
        np_d_ee,      # may have to do this one the "long way"
        np_d_mark,
        np.uintc(ecount),
        np.uintc(d_ee.size),
        block = block_dim,
        grid = grid_dim
    )
    np_d_ee.get(d_ee)
    np_d_mark.get(d_mark)
    # devdata = pycuda.tools.DeviceData()
    # orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[1])
    # logger.debug("Occupancy = %s" % (orec.occupancy * 100))
    logger.info('Finished.')
    return d_ee, d_mark


def mark_spanning_euler_edges(d_ee, d_mark , ecount,d_cg_edge,cg_edgeCount,d_tree, treeCount):
    logger = logging.getLogger(__name__)
    logger.info("started.")
    mod = SourceModule("""
    typedef unsigned long long  KEY_T ;
    typedef struct EulerVertex{
        KEY_T	vid;
        unsigned int  ep;
        unsigned int  ecount;
        unsigned int  lp;
        unsigned int  lcount;
    }EulerVertex;
    typedef struct CircuitEdge{
        unsigned int ceid;
        unsigned e1;
        unsigned e2;
        unsigned c1;
        unsigned c2;
    }CircuitEdge;
    typedef struct EulerEdge{
        KEY_T eid;
        unsigned int v1;
        unsigned int v2;
        unsigned int s;
        unsigned int pad;
    }EulerEdge;

    __global__ void  markSpanningEulerEdges(
                                            EulerEdge * ee,
                                            unsigned int * mark ,
                                            unsigned int ecount,
                                            CircuitEdge * cg_edge,
                                            unsigned int cg_edgeCount,
                                            unsigned int * tree,
                                            unsigned int treeCount)
    {

        unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
        if(tid < treeCount)
        {
            /*if(tree[tid]==1)*/{
                atomicExch(mark+min(cg_edge[tree[tid]].e1,cg_edge[tree[tid]].e2),1); // important: assumption if(mark[i]=1) means mark[i]and mark[i+1] are swipe
                //atomicExch(mark+cg_edge[tree[tid]].e2,1);

            }
        }
    }
    """)
    block_dim, grid_dim = getOptimalLaunchConfiguration(treeCount, 512)
    mark = mod.get_function('markSpanningEulerEdges')
    np_d_mark = gpuarray.to_gpu(d_mark)
    mark(
        drv.In(d_ee),
        np_d_mark,
        np.uintc(ecount),
        drv.In(d_cg_edge),
        np.uintc(cg_edgeCount),
        drv.In(d_tree),
        np.uintc(treeCount),
        block = block_dim,
        grid = grid_dim
    )
    np_d_mark.get(d_mark)
    # devdata = pycuda.tools.DeviceData()
    # orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[1])
    # logger.debug("Occupancy = %s" % (orec.occupancy * 100))
    logger.info('Finished.')
    return d_mark


def executeSwipeDevice(d_ev, d_e, vcount, d_ee, ecount, d_cg_edge, cg_edgeCount, d_tree, treeCount):
    logger = logging.getLogger('pyeulertour.executeSwipeDevice')
    logger.info("started.")
    d_mark = np.ones(ecount, dtype=np.uintc)
    # block_dim, grid_dim = getOptimalLaunchConfiguration(treeCount, 512)
    d_mark = mark_spanning_euler_edges(d_ee, d_mark, ecount, d_cg_edge, cg_edgeCount, d_tree, treeCount)
    d_ee, d_mark = execute_swipe(d_ev, d_e, vcount, d_ee, d_mark, ecount)
    logger.info('Finished')
    return d_ee


def identify_contig_start(d_ee, d_contigStart, ecount):
    logger = logging.getLogger('pyeulertour.identify_contig_start')
    logger.info("started.")
    mod = SourceModule("""
    typedef unsigned long long  KEY_T ;
    typedef struct EulerEdge{
        KEY_T eid;
        unsigned int v1;
        unsigned int v2;
        unsigned int s;
        unsigned int pad;
    }EulerEdge;

    __global__  void identifyContigStart( EulerEdge * e ,unsigned char * contigStart,unsigned int ecount){
        unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
        if(tid<ecount){
            if(e[tid].s < ecount){
                contigStart[e[tid].s]=0;
                //atomicExch(contigStart+e[tid].s,0);
            }
        }
    }

    """)
    block_dim, grid_dim = getOptimalLaunchConfiguration(ecount.item(), 512)
    np_d_contigStart = gpuarray.to_gpu(d_contigStart)
    c_start = mod.get_function('identifyContigStart')
    c_start(
        drv.In(d_ee),
        np_d_contigStart,
        np.uintc(ecount),
        block=block_dim,
        grid=grid_dim
    )
    np_d_contigStart.get(d_contigStart)
    # devdata = pycuda.tools.DeviceData()
    # orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[1])
    # logger.debug("Occupancy = %s" % (orec.occupancy * 100))
    logger.info('Finished.')
    return d_contigStart


# def markContigStart(d_ee, d_contigStart, ecount):
#     pass



def findEulerDevice(d_ev, d_l, d_e, vcount, d_ee, ecount, d_cg_edge, cg_edgeCount, cg_vertexCount):
    """

    :param d_ev:
    :param d_l:
    :param d_e:
    :param vcount:
    :param d_ee:
    :param ecount:
    :param d_cg_edge:
    :param cg_edgeCount:
    :param cg_vextexCount:
    :param kmerLength:
    :return:
    """

    # Step 1:
    # Assign successors
    d_ev, d_ee = assign_successor_device(d_ev, d_l, d_e, vcount, d_ee, ecount)

    d_v = np.zeros(ecount, dtype=[('vid', np.uintc), ('n1', np.uintc), ('n2', np.uintc)])

    d_v = construct_successor_graphP1_device(d_ee, d_v, ecount)

    d_v = construct_successor_graphP2_device(d_ee, d_v, ecount)

    d_D = np.zeros(ecount, dtype=np.uintc)
    d_D = find_component_device(d_v, d_D, ecount)
    d_C = np.zeros_like(d_D)
    # 	calculateCircuitGraphVertexData<<<grid,block>>>( d_D,d_C,ecount);
    d_D, d_C = calculate_circuit_graph_vertex_data_device(d_D, d_C, ecount)
# 	//step 4.b offset calculation .find prefix sum
# 	cudppScan(scanplan, d_cg_offset, d_C, ecount);
    d_cg_offset = np.zeros_like(d_C)
    knl = ExclusiveScanKernel(np.uintc, "a+b", 0)
    np_d_C = gpuarray.to_gpu(d_C)
    knl(np_d_C)
    np_d_C.get(d_cg_offset)

# extern "C" void readData(void * h_out, void * d_in, int length, int width) {
# cutilSafeCall(
# 		cudaMemcpy(h_out, d_in, length * width, cudaMemcpyDeviceToHost));
# }

    buffer = []
    buffer.append(d_cg_offset[(ecount - 1)])
    buffer.append(d_C[(ecount - 1)])
    circuitVertexSize = buffer[0] + buffer[1]
    cg_vertexCount = circuitVertexSize

    # allocateMemory((void **) & d_cv, circuitVertexSize * sizeof(unsigned int));
    data_size = circuitVertexSize #* UINTC #* sys.getsizeof(np.uintc)
    d_cv = np.zeros(data_size, dtype=np.uintc)
    d_cv = construct_circuit_Graph_vertex(d_C, d_cg_offset, ecount, d_cv)

    d_cedgeCount = np.zeros(ecount, dtype=np.uintc) # same C size allocated and type
    if circuitVertexSize > 1:
        d_cedgeCount = calculate_circuit_graph_edge_data(d_ev, d_e, vcount, d_D, d_cg_offset, ecount, d_cedgeCount)

        d_cg_edge_start = np.zeros_like(d_cedgeCount)
        np_d_cedgeCount = gpuarray.to_gpu(d_cedgeCount)
        knl(np_d_cedgeCount)
        np_d_cedgeCount.get(d_cg_edge_start)

        buffer[0] = d_cg_edge_start[len(d_cg_edge_start) -1]   #[circuitVertexSize - 1]
        buffer[1] = d_cedgeCount[len(d_cedgeCount) - 1]   #[circuitVertexSize - 1]
        circuitGraphEdgeCount = buffer[0] + buffer[1]

        # edge_size = 5 * UINTC # sys.getsizeof(np.uintc)
        # d_cg_edge = np.zeros(edge_size, dtype=[('ceid', np.uintc), ('e1', np.uintc), ('e2', np.uintc), ('c1', np.uintc), ('c2', np.uintc)])
        # d_cg_edge = np.zeros(ecount, dtype=[('ceid', np.uintc), ('e1', np.uintc), ('e2', np.uintc), ('c1', np.uintc), ('c2', np.uintc)])
        d_cg_edge = np.zeros(circuitGraphEdgeCount, dtype=[('ceid', np.uintc), ('e1', np.uintc), ('e2', np.uintc), ('c1', np.uintc), ('c2', np.uintc)])

        d_cg_edge = assign_circuit_graph_edge_data(d_ev, d_e, vcount, d_D, d_cg_offset, ecount, d_cg_edge_start, d_cedgeCount,
                                   circuitVertexSize, d_cg_edge,circuitGraphEdgeCount)

        # h_cg_edge = np.zeros_ like(d_cg_edge)
        d_cg_edge.sort(order=['c1', 'c2'])
    return d_cg_edge, circuitGraphEdgeCount, cg_vertexCount


