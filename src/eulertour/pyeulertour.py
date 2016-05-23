""" GPU Accelerated Euler Tour """

import numpy as np
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.driver import device_attribute
import logging

module_logger = logging.getLogger('eulercuda.pyeulertour')

def assign_successor_device():
    logger = logging.getLogger('eulercuda.pyeulertour.assign_successor_device')
    logger.info("started.")
    mod = SourceModule("""
    __global__  void assignSuccessor(EulerVertex * ev,unsigned int * l, unsigned int * e, unsigned vcount, EulerEdge * ee ,unsigned int ecount)
    {
        unsigned int tid = (blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
        unsigned int eidx = 0;
        if(tid < vcount)
        {
            while(eidx < ev[tid].ecount && eidx < ev[tid].lcount)
            {
                ee[e[ev[tid].ep + eidx]].s = l[ev[tid].lp + eidx] ;
                eidx++;
            }
        }
    }
    """)

    assign_successor = mod.get_function("assignSuccessor")



def construct_successor_graphP1_device():
    logger = logging.getLogger('eulercuda.pyeulertour.construct_successor_graphP1_device')
    logger.info("started.")
    mod = SourceModule("""
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


def construct_successor_graphP2_device():
    logger = logging.getLogger('eulercuda.pyeulertour.construct_successor_graphP1_device')
    logger.info("started.")
    mod = SourceModule("""
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


def calculate_circuit_graph_vertex_data_device():
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