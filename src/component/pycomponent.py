
import sys
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import ctypes
import logging

import encoder.pyencode as enc

# ULONGLONG = 8
# UINTC = 4

def component_step_init(d_v, d_D, d_Q, length):
    """

    :param d_v:
    :param d_Q:
    :param length:
    :return:
    """
    logger = logging.getLogger('eulercuda.pycomponent.component_step_init')
    logger.info("started.")
    mod = SourceModule("""
    typedef struct Vertex
    {
        unsigned int vid;
        unsigned int n1;
        unsigned int n2;
    } Vertex;

    __global__ void componentStepInit(Vertex * v, unsigned int * D,  unsigned int* Q, unsigned int length)
    {
        unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
        if( tid <length)
        {
            //v[tid].vid;
            D[tid]=tid;
            Q[tid]=0;
        }
    }
    """)
    component_step_init_device = mod.get_function('componentStepInit')
    block_dim, grid_dim = enc.getOptimalLaunchConfiguration(length, 512)
    np_d_D = gpuarray.to_gpu(d_D)
    np_d_Q = gpuarray.to_gpu(d_Q)
    logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    component_step_init_device(
        drv.In(d_v),
        np_d_D,
        np_d_Q,
        np.uintc(length),
        block=block_dim, grid=grid_dim
    )
    np_d_D.get(d_D)
    np_d_Q.get(d_Q)
    devdata = pycuda.tools.DeviceData()
    orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[0])
    logger.info("Occupancy = %s" % (orec.occupancy * 100))

    logger.info("Finished. Leaving.")
    return d_D, d_Q

def component_step1_shortcutting_p1(d_v, d_prevD, d_D, d_Q, length, s):
    """

    :param d_v:
    :param d_prevD:
    :param d_D:
    :param d_Q:
    :param length:
    :param s:
    :return:
    """

    logger = logging.getLogger('eulercuda.pycomponent.component_step1_shortcutting_p1')
    logger.info("started.")
    mod = SourceModule("""
     typedef struct Vertex
     {
         unsigned int vid;
         unsigned int n1;
         unsigned int n2;
     } Vertex;
    __global__ void componentStepOne_ShortCuttingP1(Vertex * v, unsigned  int * prevD, unsigned  int * curD, unsigned int * Q, unsigned int length, int s)
    {
        unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
        if( tid <length)
        {
            curD[tid] =prevD[prevD[tid]];
        }
    }

     """)

    block_dim, grid_dim = enc.getOptimalLaunchConfiguration(length, 512)
    logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    np_d_v = gpuarray.to_gpu(d_v)
    np_d_D = gpuarray.to_gpu(d_D)
    np_d_prevD = gpuarray.to_gpu(d_prevD)
    np_d_Q = gpuarray.to_gpu(d_Q)
    shortcutting_p1_device = mod.get_function('componentStepOne_ShortCuttingP1')
    shortcutting_p1_device(
        np_d_v,
        np_d_prevD,
        np_d_D,
        np_d_Q,
        np.uintc(length),
        np.uintc(s),
        block=block_dim, grid=grid_dim
    )
    np_d_v.get(d_v)
    np_d_prevD.get(d_prevD)
    np_d_D.get(d_D)
    np_d_Q.get(d_Q)
    devdata = pycuda.tools.DeviceData()
    orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[0])
    logger.info("Occupancy = %s" % (orec.occupancy * 100))

    logger.info("Finished. Leaving.")
    return d_D


def component_step1_shortcutting_p2(d_v, d_prevD, d_D, d_Q, length, s):
    """

    :param d_v:
    :param d_prevD:
    :param d_D:
    :param d_Q:
    :param length:
    :param s:
    :return:
    """


    logger = logging.getLogger('eulercuda.pycomponent.component_step1_shortcutting_p2')
    logger.info("started.")
    mod = SourceModule("""
       typedef struct Vertex
       {
           unsigned int vid;
           unsigned int n1;
           unsigned int n2;
       } Vertex;
    __global__ void componentStepOne_ShortCuttingP2(Vertex * v, unsigned  int * prevD, unsigned  int * curD, unsigned int * Q, unsigned int length, int s)
    {
        unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
        if( tid <length)
        {
            if(curD[tid]!=prevD[tid])
            {
                Q[curD[tid]]=s;
            }
        }
    }
       """)
    block_dim, grid_dim = enc.getOptimalLaunchConfiguration(length, 512)
    logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    np_d_v = gpuarray.to_gpu(d_v)
    np_d_D = gpuarray.to_gpu(d_D)
    np_d_prevD = gpuarray.to_gpu(d_prevD)
    np_d_Q = gpuarray.to_gpu(d_Q)
    shortcutting_p1_device = mod.get_function('componentStepOne_ShortCuttingP2')
    shortcutting_p1_device(
        np_d_v,
        np_d_prevD,
        np_d_D,
        np_d_Q,
        np.uintc(length),
        np.uintc(s),
        block=block_dim, grid=grid_dim
    )
    np_d_v.get(d_v)
    np_d_prevD.get(d_prevD)
    np_d_D.get(d_D)
    np_d_Q.get(d_Q)
    devdata = pycuda.tools.DeviceData()
    orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[0])
    logger.info("Occupancy = %s" % (orec.occupancy * 100))

    logger.info("Finished. Leaving.")
    return d_Q

def component_Step2_P1(d_v, d_prevD, d_D, d_Q, d_t1, d_val1, d_t2, d_val2, length, s):
    """

    :param d_v:
    :param d_prevD:
    :param d_D:
    :param d_Q:
    :param d_t1:
    :param d_val1:
    :param d_t2:
    :param d_val2:
    :param length:
    :param s:
    :return:
    """
    logger = logging.getLogger('eulercuda.pycomponent.component_Step2_P1')
    logger.info("started.")
    mod = SourceModule("""
       typedef struct Vertex
       {
           unsigned int vid;
           unsigned int n1;
           unsigned int n2;
       } Vertex;

       __global__ void componentStepTwoP1(Vertex * v,  unsigned int * prevD,  unsigned int * curD, unsigned int * Q,
            unsigned int * t1,unsigned int *val1 ,unsigned int * t2,unsigned int * val2, unsigned int length, unsigned  int s)
       {
        unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;

        if( tid <length )
        {
            t1[tid]=length;t2[tid]=length;
            //it will done for each edge 1
            if(curD[tid] == prevD[tid] && v[tid].n1<length )
            {
                if(curD[v[tid].n1] < curD[tid])
                {
                    t1[tid]=curD[tid];
                    val1[tid]=curD[v[tid].n1];

                }
            }

            //it will done for each edge 2
            if(curD[tid] == prevD[tid] && v[tid].n2<length)
            {
                if(curD[v[tid].n2] < curD[tid])
                {
                    t2[tid]=curD[tid];
                    val2[tid]=curD[v[tid].n2];
                }
            }

        }
    }

    """)
    block_dim, grid_dim = enc.getOptimalLaunchConfiguration(length, 512)
    logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    np_d_t1 = gpuarray.to_gpu(d_t1)
    np_d_t2 = gpuarray.to_gpu(d_t2)
    np_d_val1 = gpuarray.to_gpu(d_val1)
    np_d_val2 = gpuarray.to_gpu(d_val2)
    step2_P1 = mod.get_function('componentStepTwoP1')
    step2_P1(
        drv.In(d_v),
        drv.In(d_prevD),
        drv.In(d_D),
        drv.In(d_Q),
        np_d_t1,
        np_d_val1,
        np_d_t2,
        np_d_val2,
        np.uintc(length),
        np.uintc(s),
        block=block_dim, grid=grid_dim
    )
    np_d_t1.get(d_t1)
    np_d_t2.get(d_t2)
    np_d_val1.get(d_val1)
    np_d_val2.get(d_val2)
    devdata = pycuda.tools.DeviceData()
    orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[0])
    logger.info("Occupancy = %s" % (orec.occupancy * 100))

    logger.info("Finished. Leaving.")
    return d_t1, d_t2, d_val1, d_val2


def component_Step2_P2(d_v, d_prevD, d_D, d_Q, d_t1, d_val1, d_t2, d_val2, length, s):
    """

    :param d_v:
    :param d_prevD:
    :param d_D:
    :param d_Q:
    :param d_t1:
    :param d_val1:
    :param d_t2:
    :param d_val2:
    :param length:
    :param s:
    :return:
    """
    logger = logging.getLogger('eulercuda.pycomponent.component_Step2_P2')
    logger.info("started.")
    mod = SourceModule("""
       typedef struct Vertex
       {
           unsigned int vid;
           unsigned int n1;
           unsigned int n2;
       } Vertex;
    __global__ void componentStepTwoP2(Vertex * v,  unsigned int * prevD,  unsigned int * curD, unsigned int * Q,
            unsigned int * t1,unsigned int *val1 ,unsigned int * t2,unsigned int * val2, unsigned int length, unsigned  int s)
    {
        unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;

        int a;
        int val;

        if( tid <length )
        {
            //it will done for each edge 1
            if(t1[tid]<length)
            {
                a=t1[tid];
                val=val1[tid];
                atomicMin(curD+a,val);
                atomicExch(Q+val,s);

            }

            //it will done for each edge 2
            if(t2[tid]<length)
            {
                a=t2[tid];
                val=val2[tid];
                atomicMin(curD+a,val);
                atomicExch(Q+val,s);
            }
        }
    }

    """)
    block_dim, grid_dim = enc.getOptimalLaunchConfiguration(length, 512)
    logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    np_d_D = gpuarray.to_gpu(d_D)
    np_d_Q = gpuarray.to_gpu(d_Q)
    np_d_t1 = gpuarray.to_gpu(d_t1)
    np_d_t2 = gpuarray.to_gpu(d_t2)
    np_d_val1 = gpuarray.to_gpu(d_val1)
    np_d_val2 = gpuarray.to_gpu(d_val2)
    step2_P2 = mod.get_function('componentStepTwoP2')
    step2_P2(
        drv.In(d_v),
        drv.In(d_prevD),
        np_d_D,
        np_d_Q,
        np_d_t1,
        np_d_val1,
        np_d_t2,
        np_d_val2,
        np.uintc(length),
        np.uintc(s),
        block=block_dim, grid=grid_dim
    )
    np_d_D.get(d_D)
    np_d_Q.get(d_Q)
    np_d_t1.get(d_t1)
    np_d_t2.get(d_t2)
    np_d_val1.get(d_val1)
    np_d_val2.get(d_val2)
    devdata = pycuda.tools.DeviceData()
    orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[0])
    logger.info("Occupancy = %s" % (orec.occupancy * 100))

    logger.info("Finished. Leaving.")
    return d_D, d_Q


def component_Step3_P1(d_v, d_prevD, d_D, d_Q, d_t1, d_val1, d_t2, d_val2, length, s):
    """

    :param d_v:
    :param d_prevD:
    :param d_D:
    :param d_Q:
    :param d_t1:
    :param d_val1:
    :param d_t2:
    :param d_val2:
    :param length:
    :param s:
    :return:
    """
    logger = logging.getLogger('eulercuda.pycomponent.component_Step3_P1')
    logger.info("started.")
    mod = SourceModule("""
        typedef struct Vertex
        {
            unsigned int vid;
            unsigned int n1;
            unsigned int n2;
        } Vertex;

    __global__ void componentStepThreeP1(Vertex * v, unsigned int * prevD,unsigned  int * curD,unsigned int * Q,unsigned int * t1,unsigned int *val1 ,unsigned int * t2,unsigned int * val2,unsigned int length,unsigned int s){
        unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
        if( tid< length) {
            t1[tid]=length; t2[tid]=length;
            //it will be done for each edge 1
            if(curD[tid]==curD[curD[tid]] && Q[curD[tid]] < s && v[tid].n1<length){
                if( curD[tid] != curD[v[tid].n1] ){
                    t1[tid]=curD[tid];
                    val1[tid]= curD[v[tid].n1];
                }
            }
            //it will be done for each edge 2
            if(curD[tid]==curD[curD[tid]] && Q[curD[tid]] < s && v[tid].n2<length){
                if( curD[tid] != curD[v[tid].n2] ){
                    t2[tid]=curD[tid];
                    val2[tid]= curD[v[tid].n2];
                }
            }

        }
    }


        """)
    block_dim, grid_dim = enc.getOptimalLaunchConfiguration(length, 512)
    logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    np_d_t1 = gpuarray.to_gpu(d_t1)
    np_d_t2 = gpuarray.to_gpu(d_t2)
    np_d_val1 = gpuarray.to_gpu(d_val1)
    np_d_val2 = gpuarray.to_gpu(d_val2)
    step3_P1 = mod.get_function('componentStepThreeP1')
    step3_P1(
        drv.In(d_v),
        drv.In(d_prevD),
        drv.In(d_D),
        drv.In(d_Q),
        np_d_t1,
        np_d_val1,
        np_d_t2,
        np_d_val2,
        np.uintc(length),
        np.uintc(s),
        block=block_dim, grid=grid_dim
    )
    np_d_t1.get(d_t1)
    np_d_t2.get(d_t2)
    np_d_val1.get(d_val1)
    np_d_val2.get(d_val2)
    devdata = pycuda.tools.DeviceData()
    orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[0])
    logger.info("Occupancy = %s" % (orec.occupancy * 100))

    logger.info("Finished. Leaving.")
    return d_t1, d_t2, d_val1, d_val2


def component_Step3_P2(d_v, d_prevD, d_D, d_Q, d_t1, d_val1, d_t2, d_val2, length, s):
    """

    :param d_v:
    :param d_prevD:
    :param d_D:
    :param d_Q:
    :param d_t1:
    :param d_val1:
    :param d_t2:
    :param d_val2:
    :param length:
    :param s:
    :return:
    """
    logger = logging.getLogger('eulercuda.pycomponent.component_Step3_P2')
    logger.info("started.")
    mod = SourceModule("""
        typedef struct Vertex
        {
            unsigned int vid;
            unsigned int n1;
            unsigned int n2;
        } Vertex;
    __global__ void componentStepThreeP2(Vertex * v, unsigned int * prevD,unsigned  int * curD,unsigned int * Q,unsigned int * t1,unsigned int *val1 ,unsigned int * t2,unsigned int * val2,unsigned int length,unsigned int s){
        unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
        int a;
        int val;
        if( tid< length) {
            //it will be done for each edge 1
            if(t1[tid]<length){
                a=t1[tid];
                val= val1[tid];
                atomicMin(curD+a,val);

            }
            //it will be done for each edge 2
            if(t2[tid]<length){
                a=t2[tid];
                val= val2[tid];
                atomicMin(curD+a,val);

            }
        }
    }
    """)
    block_dim, grid_dim = enc.getOptimalLaunchConfiguration(length, 512)
    logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    np_d_D = gpuarray.to_gpu(d_D)
    np_d_t1 = gpuarray.to_gpu(d_t1)
    np_d_t2 = gpuarray.to_gpu(d_t2)
    np_d_val1 = gpuarray.to_gpu(d_val1)
    np_d_val2 = gpuarray.to_gpu(d_val2)
    step3_P2 = mod.get_function('componentStepThreeP2')
    step3_P2(
        drv.In(d_v),
        drv.In(d_prevD),
        np_d_D,
        drv.In(d_Q),
        np_d_t1,
        np_d_val1,
        np_d_t2,
        np_d_val2,
        np.uintc(length),
        np.uintc(s),
        block=block_dim, grid=grid_dim
    )
    np_d_D.get(d_D)
    np_d_t1.get(d_t1)
    np_d_t2.get(d_t2)
    np_d_val1.get(d_val1)
    np_d_val2.get(d_val2)
    devdata = pycuda.tools.DeviceData()
    orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[0])
    logger.info("Occupancy = %s" % (orec.occupancy * 100))

    logger.info("Finished. Leaving.")
    return d_D

def component_step4_P1(d_v, d_D, d_val1, length):
    """

    :param d_v:
    :param d_D:
    :param d_val1:
    :param length:
    :return:
    """
    logger = logging.getLogger('eulercuda.pycomponent.component_Step4_P1')
    logger.info("started.")
    mod = SourceModule("""
         typedef struct Vertex
         {
             unsigned int vid;
             unsigned int n1;
             unsigned int n2;
         } Vertex;

    __global__ void componentStepFourP1(Vertex * v, unsigned  int * curD,unsigned int * val1,unsigned int length){
        unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
        if( tid < length){
            val1[tid]=curD[curD[tid]];
        }
    }

    """)
    block_dim, grid_dim = enc.getOptimalLaunchConfiguration(length, 512)
    logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    np_d_val1 = gpuarray.to_gpu(d_val1)
    step4_P1 = mod.get_function('componentStepFourP1')
    step4_P1(
        drv.In(d_v),
        drv.In(d_D),
        np_d_val1,
        np.uintc(length),
        block=block_dim, grid=grid_dim
    )
    np_d_val1.get(d_val1)
    devdata = pycuda.tools.DeviceData()
    orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[0])
    logger.info("Occupancy = %s" % (orec.occupancy * 100))

    logger.info("Finished. Leaving.")
    return d_val1


def component_step4_P2(d_v, d_D, d_val1, length):
    """

    :param d_v:
    :param d_D:
    :param d_val1:
    :param length:
    :return:
    """
    logger = logging.getLogger('eulercuda.pycomponent.component_Step4_P2')
    logger.info("started.")
    mod = SourceModule("""
         typedef struct Vertex
         {
             unsigned int vid;
             unsigned int n1;
             unsigned int n2;
         } Vertex;

    __global__ void componentStepFourP2(Vertex * v, unsigned  int * curD,unsigned int * val1,unsigned int length){
        unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
        if( tid < length){
            curD[tid]= val1[tid];
        }

    }
    """)
    block_dim, grid_dim = enc.getOptimalLaunchConfiguration(length, 512)
    logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    np_d_val1 = gpuarray.to_gpu(d_val1)
    np_d_D = gpuarray.to_gpu(d_D)
    step4_P2 = mod.get_function('componentStepFourP2')
    step4_P2(
        drv.In(d_v),
        np_d_D,
        np_d_val1,
        np.uintc(length),
        block=block_dim, grid=grid_dim
    )
    np_d_D.get(d_D)
    np_d_val1.get(d_val1)
    devdata = pycuda.tools.DeviceData()
    orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[0])
    logger.info("Occupancy = %s" % (orec.occupancy * 100))

    logger.info("Finished. Leaving.")
    return d_D


def component_step5(d_Q,length,d_sptemp,s):
    """

    :param d_Q:
    :param length:
    :param d_sptemp:
    :param s:
    :return:
    """

    logger = logging.getLogger('eulercuda.pycomponent.component_Step5')
    logger.info("started.")
    mod = SourceModule("""
    __global__ void componentStepFive(unsigned int * Q,unsigned int length,unsigned  int * sprimtemp,unsigned int s){
        unsigned int tid=(blockDim.x*blockDim.y * gridDim.x*blockIdx.y) + (blockDim.x*blockDim.y*blockIdx.x)+(blockDim.x*threadIdx.y)+threadIdx.x;
        if(tid <length) {
            if(Q[tid]==s){
                atomicExch(sprimtemp,1);
                //*sprime=*sprimtemp+1;
            }
        }
    }
    """)
    block_dim, grid_dim = enc.getOptimalLaunchConfiguration(length, 512)
    logger.info('block_dim = %s, grid_dim = %s' % (block_dim, grid_dim))
    np_d_sptemp = gpuarray.to_gpu(d_sptemp)
    step5 = mod.get_function('componentStepFive')
    step5(
        drv.In(d_Q),
        np.uintc(length),
        np_d_sptemp,
        np.uintc(s),
        block=block_dim, grid=grid_dim
    )
    np_d_sptemp.get(d_sptemp)
    devdata = pycuda.tools.DeviceData()
    orec = pycuda.tools.OccupancyRecord(devdata, block_dim[0] * grid_dim[0])
    logger.info("Occupancy = %s" % (orec.occupancy * 100))

    logger.info("Finished. Leaving.")
    return d_sptemp


def find_component_device(d_v, d_D,  length):
    """

    :param d_v:
    :param d_D:
    :param ecount:
    :return:
    """
    logger = logging.getLogger('eulercuda.pycomponent.find_component_device')
    logger.info("started.")
    mem_size = length
    d_prevD = np.zeros(mem_size, dtype=np.uintc)
    d_Q = np.zeros_like(d_prevD)
    d_t1 = np.zeros_like(d_prevD)
    d_t2 = np.zeros_like(d_prevD)
    d_val1 = np.zeros_like(d_prevD)
    d_val2 = np.zeros_like(d_prevD)
    sp = np.uintc(0)

    s = np.uintc

    d_D, d_Q = component_step_init(d_v, d_D, d_Q, length)
    s, sp = 1, 1

    sptemp = drv.pagelocked_zeros(4, dtype=np.intc, mem_flags=drv.host_alloc_flags.DEVICEMAP)
    d_sptemp = np.intp(sptemp.base.get_device_pointer())

    while s == sp:
        d_D, d_prevD = d_prevD, d_D

        d_D = component_step1_shortcutting_p1(d_v, d_prevD, d_D, d_Q, length, s)

        d_Q = component_step1_shortcutting_p2(d_v, d_prevD, d_D, d_Q, length, s)

        d_t1, d_t2, d_val1, d_val2 = component_Step2_P1(d_v, d_prevD, d_D, d_Q, d_t1, d_val1, d_t2, d_val2, length, s)

        d_D, d_Q = component_Step2_P2(d_v, d_prevD, d_D, d_Q, d_t1, d_val1, d_t2, d_val2, length, s)

        d_t1, d_t2, d_val1, d_val2 = component_Step3_P1(d_v, d_prevD, d_D, d_Q, d_t1, d_val1, d_t2, d_val2, length, s)

        d_D = component_Step3_P2(d_v, d_prevD, d_D, d_Q, d_t1, d_val1, d_t2, d_val2, length, s)

        d_val1 = component_step4_P1(d_v, d_D, d_val1, length)

        d_D = component_step4_P2(d_v, d_D, d_val1, length)

        sptemp[0] = 0

        d_sptemp = (d_Q, length, d_sptemp, s)

        sp += sptemp[0]

        s += 1

    logger.info("Finished. Leaving.")
    return d_D


