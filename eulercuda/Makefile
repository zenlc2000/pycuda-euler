################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE      := eulercuda

# CUDA source files (compiled with cudacc)
CUFILES_sm_13        := debruijn.cu eulertour.cu component.cu gpuhash.cu gpuhash2.cu  encoder.cu fixer.cu 

# CUDA dependency files
CU_DEPS         := \
        Kmer.h Graph.h utils.h cudpp.h common.h gpuhash_device.h  gpuhash_device2.h stats.h encoder.h fixer.cuh

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= FastaReader.cpp main.cpp ReadBuffer.cpp Sequence.cpp utils.cpp stats.cpp  fixer.cpp path.cpp transform.cpp

COMMON_FLAGS	:=-DLEMON_ENABLE_DEBUG 

BOOST_LIB	:= /usr/lib64
#BOOST_LIB	:= /home/smahmoo4/local/tools/boost_1_43_0/stage/lib/
BOOST_DIR	:= /usr/include/boost/
#BOOST_DIR	:= /home/smahmoo4/local/tools/boost_1_43_0/boost/
SPARSE_HASH_INC	:= /tmp/faraz/local/include

INCLUDES	+= -I$(BOOST_DIR) -I$(SPARSE_HASH_INC) 
GPUEULER_LIB	:= -L$(BOOST_LIB) 

USECUDPP	:=1
USEDRVAPI	:=1
#dbg		:=1
#keep		:=1
verbose		:=1

################################################################################
# Rules and targets

include ./common/common.mk


