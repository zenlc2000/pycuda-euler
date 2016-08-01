
#include <cutil_inline.h>
//#include <google/dense_hash_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "utils.h"
#include "stats.h"
#include "gpuhash.h"
#include "gpuhash_device.h"
#include "Graph.h"
//#include <boost/foreach.hpp>

#define NA_COUNT 4

__device__ __constant__ char mutator[8][4] = {
    { 0, 0, 0, 0}, //normal
    { 1, 1, 1, 1},
    { 2, 2, 2, 2},
    { 3, 3, 3, 3},
    { 0, 1, 2, 3}, //mutated
    { 1, 2, 3, 0},
    { 2, 3, 0, 1},
    { 3, 0, 1, 2}
};
__device__ __constant__ char base4[] = {0, 0, 0, 1, 3, 0, 0, 2};
__device__ __constant__ char reverse[] = {3, 2, 1, 0};
__device__ __constant__ unsigned int code[] = {65, 67, 71, 84};

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
/* 	Computes Mutation for a read
 //d_read,readLength,l,M
 d_TK,d_TV,d_bucketSize,bucketCount,d_mutation
 * 	called as <<readLength,4,l>>
 *
 * */
__global__ void calculateMutationScores(char * in_read, char * mutationMask,
        unsigned int readLength, unsigned int lmerLength, unsigned int M,
        KEY_PTR TK, VALUE_PTR TV, unsigned int * bucketSize,
        unsigned int bucketCount,
        /*out param*/
        unsigned int* mutation, //out
        unsigned int * buffer) {

    extern __shared__ char read[];
    char myMutationMask = 0;
    VALUE_T statusF;
    VALUE_T statusR;
    KEY_T lmer = 0;
    KEY_T temp = 0;

    read[threadIdx.x] = in_read[threadIdx.x]; //simple copy
    __syncthreads();
    read[threadIdx.x] = base4[read[threadIdx.x] & 0x07]; //base4 covnert

    __syncthreads();
    myMutationMask = mutationMask[(threadIdx.x + readLength - blockIdx.y) % readLength ]; //shifted

    read[threadIdx.x] = read[threadIdx.x] | myMutationMask; //mutated mask
    __syncthreads();
    read[threadIdx.x] = mutator[read[threadIdx.x] ][blockIdx.x]; //mutated
    __syncthreads();

    for (unsigned int i = 0; i < lmerLength; i++) { //calculate lmer
        lmer = lmer << 2;
        lmer = lmer | ((KEY_T) read[(threadIdx.x + i) % readLength]); //wraping for dummy entries
    };
    //getLmerFromHash
    statusF = getHashValue(lmer, TK, TV, bucketSize, bucketCount);
    //statusF=code[read[threadIdx.x]];
    buffer[blockIdx.y * (blockDim.x * gridDim.x) + blockDim.x * blockIdx.x
            + threadIdx.x] = (unsigned int) (statusF);
    if (statusF == MAX_INT)
        statusF = 0x0;
    if (statusF >= M)
        statusF = 1;
    else statusF = 0;
    //complement

    temp = 0;
    lmer = 0;



    read[threadIdx.x] = reverse[read[threadIdx.x]];
    __syncthreads();
    for (unsigned int i = 0; i < lmerLength; i++) {
        temp = ((KEY_T) read[(threadIdx.x + i) % blockDim.x]);
        lmer = (temp << (i << 1)) | lmer;
    }

    //getLmerFromHash
    statusR = getHashValue(lmer, TK, TV, bucketSize, bucketCount);
    if (statusR == MAX_INT)
        statusR = 0;
    if (statusR >= M)
        statusR = 1;
    else statusR = 0;
    //increment solidCount
    mutation[blockIdx.y * (blockDim.x * gridDim.x) + blockDim.x * blockIdx.x
            + threadIdx.x] = (unsigned int) (statusF + statusR);

}


// l threads / per block

__global__ void calculateMutationScores2(char * in_read, char * mutationMask,
        unsigned int readLength, unsigned int lmerLength, unsigned int M,
        KEY_PTR TK, VALUE_PTR TV, unsigned int * bucketSize,
        unsigned int bucketCount,
        /*out param*/
        unsigned int* mutation, //out
        unsigned int * buffer) {

    extern __shared__ char read[];
    char * mask = read + readLength + 31;
    char myMutationMask = 0;
    VALUE_T statusF;
    VALUE_T statusR;
    KEY_T lmer = 0;

    read[threadIdx.x] = base4[ in_read[threadIdx.x] & 0x07]; //base4 covnert
    mask[threadIdx.x] = mutationMask[threadIdx.x]; //readMask
    __syncthreads();

    /*
     *read[threadIdx.x] = read[threadIdx.x] | myMutationMask; //mutated mask
            __syncthreads();
            read[threadIdx.x] = mutator[read[threadIdx.x] ][blockIdx.x]; //mutated
            __syncthreads();
     * */
    for (unsigned int j = 0; j < readLength; j++) {

        for (unsigned k = 0; k < 4; k++) {
            lmer = 0;
            for (unsigned int i = 0; i < 32; i++) { //calculate lmer

                lmer = (lmer << 2) | ((KEY_T) mutator[mask[threadIdx.x+i]|read[threadIdx.x+i]][k]); //wraping for dummy entries
            };
            lmer = (lmer >> ((32 - lmerLength) << 1)) & lmerMask[lmerLength-1];
            //getLmerFromHash
            statusF = getHashValue(lmer, TK, TV, bucketSize, bucketCount);
            buffer[blockIdx.x*(readLength*NA_COUNT*readLength) + j*readLength*NA_COUNT + k *readLength + threadIdx.x] =statusF;
            //statusF=code[read[threadIdx.x]];            
            statusF = ((statusF == MAX_INT) ? 0 : ((statusF >= M)) ? 1 : 0);


	 //blockIdx.x*(readLength*NA_COUNT*readLength) + j*readLength*NA_COUNT + k *readLength + threadIdx.x;// (unsigned int) (statusF);

            //complement
            lmer = 0;
            for ( int i = 31; i >= 0; i--) {
                lmer = (lmer << 2) | ((KEY_T) reverse[mutator[mask[threadIdx.x+i]|read[threadIdx.x+i]][k]]); //wraping for dummy entries            
            }
            lmer = (lmer) & lmerMask[lmerLength-1];
            
            //getLmerFromHash
            statusR = getHashValue(lmer, TK, TV, bucketSize, bucketCount);
            statusR = ((statusR == MAX_INT) ? 0 : ((statusR >= M)) ? 1 : 0);

            //increment solidCount
            mutation[blockIdx.x*(readLength*NA_COUNT*readLength) + j*readLength*NA_COUNT + k *readLength + threadIdx.x] = (unsigned int) (statusF + statusR);
//            buffer[blockIdx.x*(readLength*NA_COUNT*readLength) + j*readLength*NA_COUNT + k *readLength + threadIdx.x] = (unsigned int) (statusF);

        }
        //shift mask
        myMutationMask = mask[threadIdx.x ];
        __syncthreads();

        mask[(threadIdx.x+1) %readLength] = myMutationMask; //shifted
        __syncthreads();
    }
}


__global__ void calculateMutationScores3(char * in_read, char * mutationMask,
        unsigned int readLength, unsigned int lmerLength, unsigned int M,
        KEY_PTR TK, VALUE_PTR TV, unsigned int * bucketSize,
        unsigned int bucketCount,
        /*out param*/
        unsigned int* mutation, //out
        unsigned int * buffer) {

    extern __shared__ char read[];
    char * mask = read + readLength + 31;

    VALUE_T statusF;
    VALUE_T statusR;
    volatile KEY_T lmer = 0;

    read[threadIdx.x] = base4[ in_read[threadIdx.x] & 0x07]; //base4 covnert
    mask[threadIdx.x] = mutationMask[(threadIdx.x + readLength - blockIdx.y) % readLength]; //readMask
    __syncthreads();

	 for (unsigned k = 0; k < 4; k++) {
		lmer = 0;

//#pragma unroll 32
		for (unsigned int i = 0; i < 32; i++) { //calculate lmer

			lmer = (lmer << 2) | ((KEY_T) mutator[mask[threadIdx.x+i]|read[threadIdx.x+i]][k]); //wraping for dummy entries
		};
		lmer = (lmer >> ((32 - lmerLength) << 1)) & lmerMask[lmerLength-1];
		//getLmerFromHash
		statusF = getHashValue(lmer, TK, TV, bucketSize, bucketCount);
		buffer[blockIdx.x*(readLength*NA_COUNT*readLength) + blockIdx.y*readLength*NA_COUNT + k *readLength + threadIdx.x] =statusF;
		//statusF=code[read[threadIdx.x]];
		statusF = ((statusF == MAX_INT) ? 0 : ((statusF >= M)) ? 1 : 0);




		//complement
		lmer = 0;
//#pragma unroll 32
		for ( int i = 31; i >= 0; i--) {
			lmer = (lmer << 2) | ((KEY_T) reverse[mutator[mask[threadIdx.x+i]|read[threadIdx.x+i]][k]]); //wraping for dummy entries
		}
		lmer = (lmer) & lmerMask[lmerLength-1];

		//getLmerFromHash
		statusR = getHashValue(lmer, TK, TV, bucketSize, bucketCount);
		statusR = ((statusR == MAX_INT) ? 0 : ((statusR >= M)) ? 1 : 0);

		//increment solidCount
		mutation[blockIdx.x*(readLength*NA_COUNT*readLength) + blockIdx.y*readLength*NA_COUNT + k *readLength + threadIdx.x] = (unsigned int) (statusF + statusR);


	}
	//shift mask


}

__global__ void calculateMutationScores4(char * in_read, char * mutationMask,
        unsigned int readLength, unsigned int lmerLength, unsigned int M,
        KEY_PTR TK, VALUE_PTR TV, unsigned int * bucketSize,
        unsigned int bucketCount,
        /*out param*/
        unsigned int* mutation, //out
        unsigned int * buffer) {

    extern __shared__ char read[];
    char * mask = read + readLength + 31;

    VALUE_T statusF;
    VALUE_T statusR;
    volatile KEY_T lmer = 0;

    read[threadIdx.x] = base4[ in_read[blockIdx.x*readLength+threadIdx.x] & 0x07]; //base4 covnert
    mask[threadIdx.x] = mutationMask[(threadIdx.x + readLength - blockIdx.y) % readLength]; //readMask
    __syncthreads();

	 for (unsigned int k = 0; k < 4; k++) {
		lmer = 0;

#pragma unroll 4
		for (unsigned int i = 0; i < 8; i++) { //calculate lmer


		lmer= (lmer<< 8) |	((KEY_T)(shifter[mutator[mask[threadIdx.x+i*4]|read[threadIdx.x+i*4]][k]][3] |
								shifter[mutator[mask[threadIdx.x+i*4+1]|read[threadIdx.x+i*4+1]][k]][2] |
								shifter[mutator[mask[threadIdx.x+i*4+2]|read[threadIdx.x+i*4+2]][k]][1] |
								mutator[mask[threadIdx.x+i*4+3]|read[threadIdx.x+i*4+3]][k] ) ) ;
		};
		lmer = (lmer >> ((32 - lmerLength) << 1)) & lmerMask[lmerLength-1];
		//getLmerFromHash
		//statusF=20;
		statusF = getHashValue(lmer, TK, TV, bucketSize, bucketCount);
		//buffer[blockIdx.x*(readLength*NA_COUNT*readLength) + blockIdx.y*readLength*NA_COUNT + k *readLength + threadIdx.x] =statusF;
		//statusF=code[read[threadIdx.x]];
		statusF = ((statusF == MAX_INT) ? 0 : ((statusF >= M)) ? 1 : 0);
		//complement
		lmer = 0;
#pragma unroll 4
		for ( int i = 7; i >= 0; i--) {

			lmer= (lmer<<8) | ((KEY_T)
					shifter[reverse[mutator[mask[threadIdx.x+i*4+3]|read[threadIdx.x+i*4+3]][k]]][3] |
					shifter[reverse[mutator[mask[threadIdx.x+i*4+2]|read[threadIdx.x+i*4+2]][k]]][2] |
					shifter[reverse[mutator[mask[threadIdx.x+i*4+1]|read[threadIdx.x+i*4+1]][k]]][1] |
					reverse[mutator[mask[threadIdx.x+i*4]|read[threadIdx.x+i*4]][k]]

					);
		}
		lmer = (lmer) & lmerMask[lmerLength-1];

		//getLmerFromHash
		//statusR=20;
		statusR = getHashValue(lmer, TK, TV, bucketSize, bucketCount);
		statusR = ((statusR == MAX_INT) ? 0 : ((statusR >= M)) ? 1 : 0);

		//increment solidCount
		mutation[blockIdx.x*(readLength*NA_COUNT*readLength) + blockIdx.y*readLength*NA_COUNT + k *readLength + threadIdx.x] = (unsigned int) (statusF + statusR);


	}

}

__global__ void accumulate(unsigned int * mutation, unsigned int readLength,
        unsigned int l,
        /*out*/
        unsigned int * mutationScore) {

    unsigned int sum = 0;
    for (int i = 0; i < readLength - l + 1; i++) {
        sum = sum
                + mutation[blockIdx.y * (readLength * gridDim.x)
                + readLength * blockIdx.x + i];
    }
    mutationScore[blockIdx.y * gridDim.x + blockIdx.x] = sum;
}

__global__ void accumulate2(unsigned int * mutation, unsigned int readLength,
        unsigned int l,
        /*out*/
        unsigned int * mutationScore) {

    unsigned int sum = 0;
    for (int i = 0; i < readLength - l + 1; i++) {
        sum = sum
                + mutation[blockIdx.y*(readLength*NA_COUNT*readLength) + blockIdx.x*readLength+ threadIdx.x *readLength*NA_COUNT + i];
        //+ mutation[blockIdx.y*(readLength*NA_COUNT*readLength) + blockIdx.x*readLength*NA_COUNT + threadIdx.x *readLength*NA_COUNT + i];
    }
    mutationScore[blockIdx.y*(NA_COUNT*readLength) + blockIdx.x+ threadIdx.x *NA_COUNT] = sum;
}

__global__ void bestMutation2(unsigned int * mutationScore, unsigned int readLength,
        unsigned int l,
        /*out*/
        unsigned int * mutationStep) {

    unsigned int bestIdx=0;
    unsigned int bestScore=mutationScore[blockIdx.x*readLength*NA_COUNT+threadIdx.x *NA_COUNT ];
    unsigned int newScore=0;
    for (int i = 1; i < NA_COUNT ; i++) {
    	newScore=mutationScore[blockIdx.x*readLength*NA_COUNT+threadIdx.x *NA_COUNT +i];
        bestIdx=bestScore<newScore? i:bestIdx;
        bestScore=bestScore<newScore ? newScore:bestScore;
    }
    mutationStep[blockIdx.x*readLength +threadIdx.x] = (bestIdx<<16 | bestScore);
}


__global__ void bestFinalMutation2(unsigned int * mutationStep, unsigned int readLength,
        unsigned int l,
        /*out*/
        unsigned int * bestMutationPos,
        unsigned int * bestMutationIdx) {

	unsigned int newScoreValue=mutationStep[blockIdx.x*readLength ];
    unsigned int bestScore=(newScoreValue & 0x0000FFFF);
    unsigned int bestIdx=newScoreValue>>16;
    unsigned int bestPos=0;


    unsigned int newScore;

    for (int i = 1; i < readLength ; i++) {
    	newScoreValue=mutationStep[blockIdx.x*readLength +i];
    	newScore=(newScoreValue & 0x0000FFFF);
        bestIdx=bestScore<newScore? (newScoreValue>>16): bestIdx;
        bestPos=bestScore<newScore ?i:bestPos;
        bestScore=bestScore<newScore ? newScore:bestScore;
    }
    //bestMutationPos[blockIdx.x] =( bestScore>= 2* (readLength-l +1) ) ? bestPos: readLength;
    bestMutationPos[blockIdx.x] =( bestScore> ((readLength-l+1)+(readLength-l+1)>>1) ) ? bestPos: readLength;
    bestMutationIdx[blockIdx.x] =bestIdx;
}

extern "C" void getMutationScores(char * d_read, char * d_mutationMask,
        unsigned int readLength, unsigned int lmerLength, unsigned int M,
        KEY_PTR d_TK, VALUE_PTR d_TV, unsigned int * d_bucketSize,
        unsigned int bucketCount,
        /*out param*/
        unsigned int * d_mutation, //out
        unsigned int * d_buffer) {
    //calculate mutation
    dim3 grid(NA_COUNT, readLength);
    dim3 block(readLength);
    calculateMutationScores <<<grid, block, readLength >>>(d_read, d_mutationMask, readLength, lmerLength, M,
            d_TK, d_TV, d_bucketSize, bucketCount, d_mutation, d_buffer);
    CheckCUDAError();
}


extern "C" void getMutationScores2(char * d_read, char * d_mutationMask,
        unsigned int readLength, unsigned int lmerLength, unsigned int M,
        KEY_PTR d_TK, VALUE_PTR d_TV, unsigned int * d_bucketSize,
        unsigned int bucketCount,
        unsigned int readCount,
        /*out param*/
        unsigned int * d_mutation, //out
        unsigned int * d_buffer) {
    //calculate mutation
    dim3 grid(readCount);
    dim3 block(readLength);
    calculateMutationScores2 <<<grid, block, (readLength+31)*2 >>>(d_read, d_mutationMask, readLength, lmerLength, M,
            d_TK, d_TV, d_bucketSize, bucketCount, d_mutation, d_buffer);
    CheckCUDAError();
}

extern "C" void getMutationScores3(char * d_read, char * d_mutationMask,
        unsigned int readLength, unsigned int lmerLength, unsigned int M,
        KEY_PTR d_TK, VALUE_PTR d_TV, unsigned int * d_bucketSize,
        unsigned int bucketCount,
        unsigned int readCount,
        /*out param*/
        unsigned int * d_mutation, //out
        unsigned int * d_buffer) {
    //calculate mutation
    dim3 grid(readCount,readLength);
    dim3 block(readLength);
    calculateMutationScores4 <<<grid, block, (readLength+31)*2 >>>(d_read, d_mutationMask, readLength, lmerLength, M,
            d_TK, d_TV, d_bucketSize, bucketCount, d_mutation, d_buffer);
    CheckCUDAError();
}

extern "C" void getAccumulatedScores(unsigned int * d_mutation,
        unsigned int readLength, unsigned int l,
        /*out*/
        unsigned int * d_mutationScore) {
    dim3 grid(NA_COUNT, readLength);
    dim3 block(1);
    accumulate <<<grid, block >>>(d_mutation, readLength, l, d_mutationScore);
    CheckCUDAError();

}
//nxnx 4 -> n x 4
extern "C" void getAccumulatedScores2(unsigned int * d_mutation,
        unsigned int readLength, unsigned int l,
        unsigned int readCount,
        /*out*/
        unsigned int * d_mutationScore) {
    //dim3 grid(NA_COUNT, readCount);
    //dim3 block(readLength);
    dim3 grid(NA_COUNT, readCount);
    dim3 block(readLength);
    accumulate2 <<<grid, block >>>(d_mutation, readLength, l, d_mutationScore);
    CheckCUDAError();

}
// n x 4 -> n
extern "C" void getBestMutation2(unsigned int * d_mutationScore,
        unsigned int readLength, unsigned int l,
        unsigned int readCount,
        /*out*/
        unsigned int * d_mutationStep) {
    dim3 grid( readCount);
    dim3 block(readLength);
    bestMutation2 <<<grid, block >>>(d_mutationScore, readLength, l, d_mutationStep);
    CheckCUDAError();

}

//n -> 1
extern "C" void getBestFinalMutation2(unsigned int * d_mutationStep,
        unsigned int readLength, unsigned int l,
        unsigned int readCount,
        /*out*/
        unsigned int * d_bestMutationPos,
        unsigned int * d_bestMutationIdx) {
    dim3 grid(readCount);
    dim3 block(1);
    bestFinalMutation2 <<<grid, block >>>(d_mutationStep, readLength, l, d_bestMutationPos,d_bestMutationIdx);
    CheckCUDAError();

}

