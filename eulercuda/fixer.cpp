#include <iostream>
#include <iterator>
#include <cutil_inline.h>
#include <google/dense_hash_map>
#include <boost/unordered_map.hpp>
#include <boost/foreach.hpp>
#include <cuda.h>
//#include <cuda_runtime.h>
#include "common.h"
#include "utils.h"
#include "stats.h"
#include "gpuhash.h"
//#include "gpuhash_device.h"
#include "Graph.h"

#include "encoder.h"
#include "fixer.cuh"

using namespace boost;
using google::dense_hash_map;
using namespace std;

#define NA_COUNT 4
#define BATCH_SIZE 1024

#ifdef EULER_NDEBUG
#define DEBUG_FIXER_CPP(x)
#else
#define DEBUG_FIXER_CPP(x) x
#endif


#define DEBUG_CALL(x) DEBUG_FIXER_CPP(x)

static char mutator[8][4]={
		{0,0,0,0},	//0
		{'A','C','G','T'},	//A
		{0,0,0,0}, //0
		{'C','G','T','A'}, //C
		{'T','A','C','A'}, //T
		{0,0,0,0}, //0
		{0,0,0,0}, //0
		{'G','T','A','C'}, //G
};

unsigned int extractLmers(const char * reads, unsigned int readLength,
		unsigned int readCount, int lmerLength,
		/* output param**/
		KEY_PTR * lmerKeys, VALUE_PTR * lmerValues) {

	char * d_reads = NULL;
	KEY_PTR h_lmersF = NULL;
	KEY_PTR h_lmersR = NULL;
	KEY_PTR d_lmers = NULL;
	unsigned int readProcessed = 0;
	//unsigned int kmerGPUEncTimer = 0;
	unsigned int lmerCount = 0;

	 //typedef unordered_map<KEY_T, VALUE_T> map;
	 typedef dense_hash_map<KEY_T, VALUE_T> map;

	 map lmerMap(readLength*readCount);

	 unsigned int lmerEmpty=0;



	 lmerMap.set_empty_key((KEY_T)(0));
	 //we need to process into blocks since we dont have enough cuda memory
	 //buffer size ( max allocatio)
	 unsigned int readToProcess=readCount<CUDA_NUM_READS?readCount:CUDA_NUM_READS;
	 unsigned int entriesCount = readLength *readToProcess; // entries count
	 unsigned int bufferSize = sizeof(char) * readLength * readToProcess; //input buffer Size
	 unsigned int ebSize = entriesCount * sizeof(KEY_T); //entries buffer Size
	 unsigned int validLmerCount = 0;
	 //alloc mem

	 h_lmersF = (KEY_PTR) malloc(ebSize);
	 h_lmersR = (KEY_PTR) malloc(ebSize);

	 //allocate cuda mem
	 allocateMemory((void**) &d_reads, bufferSize);
	 allocateMemory((void**) &d_lmers, ebSize);



	 validLmerCount = readLength - lmerLength + 1;
	 while(readProcessed<readCount){
	 //copy symem to gpumem
	 cutilSafeCall(cudaMemcpy(d_reads, (reads+(readProcessed*readLength)), bufferSize,cudaMemcpyHostToDevice));

	 //encode
	 encodeLmer(d_reads, bufferSize, readLength, d_lmers, lmerLength,entriesCount);
	 //copy result,
	 cutilSafeCall(cudaMemcpy(h_lmersF, d_lmers, ebSize, cudaMemcpyDeviceToHost));
	 encodeLmerComplement(d_reads, bufferSize, readLength, d_lmers,lmerLength, entriesCount);
	 //copy result
	 cutilSafeCall(cudaMemcpy(h_lmersR, d_lmers, ebSize, cudaMemcpyDeviceToHost));

	 //set
		for (unsigned int j = 0; j < readToProcess; j++) {
			for (unsigned int i = 0; i < validLmerCount; i++) {
				unsigned int index = j * readLength + i;
				//create map for kmer/lmer map[kmer]=readId;
				//create list for each read read[i]=h_pkmer[index];

				if (h_lmersF[index] == 0)
					lmerEmpty++;
				else
					lmerMap[h_lmersF[index]]++;
				if (h_lmersR[index] == 0)
					lmerEmpty++;
				else
					lmerMap[h_lmersR[index]]++;

			}
		}

	 readProcessed+=readToProcess;
	 readToProcess=readCount-readProcessed;
	 readToProcess=readToProcess>CUDA_NUM_READS?CUDA_NUM_READS:readToProcess;	//how many reads to process ,max NUM_CUDA
	 entriesCount = readLength *readToProcess; // entries count
	 bufferSize = sizeof(char) * readLength * readToProcess; //input buffer Size
	 ebSize = entriesCount * sizeof(KEY_T); //entries buffer Size

	 }

	 lmerCount = (unsigned int) lmerMap.size()+(lmerEmpty>0?1:0);
	 logMessage(LOG_LVL_MSG, "#l-mer count : %d", lmerCount);
	 *lmerKeys = (KEY_PTR) malloc(lmerCount * KEY_SIZE);
	 *lmerValues = (VALUE_PTR) malloc(lmerCount * VALUE_SIZE);

	 unsigned int index = 0;
	 BOOST_FOREACH(map::value_type pair, lmerMap) {
	 (*lmerKeys)[index]=pair.first;
	 (*lmerValues)[index]=pair.second;
	 index++;
	 }
	 if(lmerEmpty>0){
	 (*lmerKeys)[index]=0; //<- LmerNULL Key
	 (*lmerValues)[index]=lmerEmpty;
	 }
	 deallocateMemory(d_reads);
	 deallocateMemory(d_lmers);



	 free(h_lmersF);
	 free(h_lmersR);
	return lmerCount;
	//TODO reset device
}
//returns correctedReadCount;
extern "C"
unsigned int errorCorrection(char * in_reads,unsigned int readCount,
		unsigned int readLength,		 char ** correctedReads,unsigned int tuple_size) {

	//allocate mem
	//transfer mem

	unsigned int in_readBufferSize = readLength * readCount;
	unsigned int correctedReadCount = 0;
	unsigned int lmerCount = 0;
	unsigned int l = tuple_size;//20;
	KEY_PTR h_lmerKeys = NULL;
	VALUE_PTR h_lmerValues = NULL;
	KEY_PTR d_lmerKeys = NULL;
	VALUE_PTR d_lmerValues = NULL;

	KEY_PTR d_TK = NULL;
	VALUE_PTR d_TV = NULL;
	unsigned int * d_bucketSize = NULL;
	unsigned int bucketCount = 0;
	unsigned int tableLength = 0;

	char * d_read = NULL;
	unsigned int * d_mutation = NULL; //for single read
	unsigned int * h_mutation =NULL;
	unsigned int * d_buffer=NULL;
	unsigned int * h_buffer=NULL;
	char * h_mutationMask = NULL; //for single read
	char * d_mutationMask = NULL; //for single read
	unsigned int * d_mutationScore = NULL;
	unsigned int * h_mutationScore = NULL; // all reads data
	unsigned int * d_mutationStep=NULL;
	unsigned int * h_mutationStep=NULL;
	unsigned int * d_bestMutationPos=NULL;
	unsigned int * d_bestMutationIdx=NULL;
	unsigned int * h_bestMutationPos=NULL;
	unsigned int * h_bestMutationIdx=NULL;
	unsigned int M = 4; //coverage;
	unsigned int readMutationSize = 0;
	unsigned int readMutationScoreSize = 0;
	unsigned int * mutationResultBasePos = NULL;
	unsigned int * mutationResultValue = NULL;
	unsigned int mutatedCount = 0;
	unsigned int maxBatchSize=readCount<BATCH_SIZE?readCount:BATCH_SIZE;
	unsigned int batchSize=maxBatchSize;
	unsigned int readMutationStepSize=0;
	unsigned int readBestMutationSize=0;

	////Timers
	unsigned int mutationTimer=0;
	unsigned int accumulateTimer=0;
	unsigned int selectMutationTimer=0;
	unsigned int selectPositionTimer=0;

//	allocateMemory((void**) &d_in_reads, in_readBufferSize);
//	cutilSafeCall(cudaMemcpy(d_reads, *reads, bufferSize,cudaMemcpyHostToDevice));
	lmerCount = extractLmers(in_reads, readLength, readCount, l, &h_lmerKeys,&h_lmerValues);

	allocateMemory((void**) &d_lmerKeys, lmerCount * (KEY_SIZE));
	allocateMemory((void**) &d_lmerValues, lmerCount * (VALUE_SIZE));
	cutilSafeCall(
			cudaMemcpy(d_lmerKeys, h_lmerKeys, lmerCount * (KEY_SIZE),
					cudaMemcpyHostToDevice));
	cutilSafeCall(
			cudaMemcpy(d_lmerValues, h_lmerValues, lmerCount * (VALUE_SIZE),
					cudaMemcpyHostToDevice));
	createHashTable(d_lmerKeys, d_lmerValues, lmerCount, &d_TK, &d_TV,
			&tableLength, &d_bucketSize, &bucketCount);


	readMutationSize = batchSize* 4 * readLength * readLength * (sizeof(unsigned int));
	readMutationScoreSize = batchSize * 4 * readLength 	* (sizeof(unsigned int));
	readMutationStepSize = batchSize * readLength 	* (sizeof(unsigned int));
	readBestMutationSize= batchSize* (sizeof(unsigned int));
	allocateMemory((void**) &d_read, batchSize*readLength * (sizeof(char)));
	allocateMemory((void**) &d_mutation, readMutationSize); // 4 x l x readlength  possible


	allocateMemory((void**) &d_mutationStep, readMutationStepSize);
	allocateMemory((void**) &d_bestMutationPos, readBestMutationSize);
	allocateMemory((void**) &d_bestMutationIdx, readBestMutationSize);

	h_mutationStep=(unsigned int * )malloc(readMutationStepSize);
	h_bestMutationPos=(unsigned int * )malloc(readBestMutationSize);
	h_bestMutationIdx=(unsigned int * )malloc(readBestMutationSize);

	//TODO remove ;for DEBUG
	h_mutation=(unsigned int * )malloc(readMutationSize);
	allocateMemory((void**) &d_buffer, readMutationSize); // 4 x l x readlength  possible
	h_buffer=(unsigned int * )malloc(readMutationSize);

	allocateMemory((void**) &d_mutationMask, readLength);
	h_mutationMask = (char *) malloc(readLength * (sizeof(char)));

	//copy mask
	h_mutationMask[0] = 0x04;
	memset(h_mutationMask + 1,0, readLength - 1);
	cutilSafeCall(
			cudaMemcpy(d_mutationMask, h_mutationMask, readLength,
					cudaMemcpyHostToDevice));

	allocateMemory((void**) &d_mutationScore, readMutationScoreSize); // 4 x l x readlength  possible
	h_mutationScore = (unsigned int *) malloc(readMutationScoreSize);
	mutationResultBasePos = (unsigned int *) malloc((sizeof(unsigned int) * readCount));
	mutationResultValue = (unsigned int *) malloc((sizeof(unsigned int) * readCount));
	memset(h_mutationScore,0,readMutationScoreSize);

	cutilCheckError(cutCreateTimer(&mutationTimer));
	cutilCheckError(cutCreateTimer(&accumulateTimer));
	cutilCheckError(cutCreateTimer(&selectMutationTimer));
	cutilCheckError(cutCreateTimer(&selectPositionTimer));

	//calculateMutations()
	for (unsigned int readIndex = 0; readIndex < readCount; ) { //each read or may be batch together bunch of reads
        //copy read
        cutilSafeCall( cudaMemcpy(d_read, in_reads + ( readLength * readIndex), readLength * batchSize, cudaMemcpyHostToDevice));
        cutilCheckError(cutStartTimer(mutationTimer));
        getMutationScores3(d_read, d_mutationMask, readLength, l, M, d_TK, d_TV, d_bucketSize, bucketCount,batchSize, d_mutation, d_buffer);
        CheckCUDAError();
        cutilCheckError(cutStopTimer(mutationTimer));
        logMessage(LOG_LVL_DETAIL,"kernel: getMutationScores3");

      ///*DEBUG*/ cutilSafeCall(cudaMemcpy(h_mutation, d_mutation, readMutationSize, cudaMemcpyDeviceToHost));
        ///*DEBUG*/  cutilSafeCall(cudaMemcpy(h_buffer, d_buffer, readMutationSize, cudaMemcpyDeviceToHost));

        cutilCheckError(cutStartTimer(accumulateTimer));
        getAccumulatedScores2(d_mutation, readLength, l,batchSize, d_mutationScore);
        ///*DEBUG*/ cutilSafeCall(cudaMemcpy(h_mutationScore, d_mutationScore, readMutationScoreSize, cudaMemcpyDeviceToHost));
        CheckCUDAError();
        cutilCheckError(cutStopTimer(accumulateTimer));
        logMessage(LOG_LVL_DETAIL,"kernel: getAccumulatedScores2");

        cutilCheckError(cutStartTimer(selectMutationTimer));
		getBestMutation2(d_mutationScore, readLength, l,batchSize, d_mutationStep);
		///*DEBUG*/cutilSafeCall(cudaMemcpy(h_mutationStep, d_mutationStep, readMutationStepSize, cudaMemcpyDeviceToHost));
		CheckCUDAError();
		cutilCheckError(cutStopTimer(selectMutationTimer));
		logMessage(LOG_LVL_DETAIL,"kernel: getBestMutation2");

		cutilCheckError(cutStartTimer(selectPositionTimer));
		getBestFinalMutation2(d_mutationStep, readLength, l,batchSize, d_bestMutationPos,d_bestMutationIdx);
		///*DEBUG*/cutilSafeCall(cudaMemcpy(h_bestMutationPos, d_bestMutationPos, readBestMutationSize, cudaMemcpyDeviceToHost));
		///*DEBUG*/cutilSafeCall(cudaMemcpy(h_bestMutationIdx, d_bestMutationIdx, readBestMutationSize, cudaMemcpyDeviceToHost));
		CheckCUDAError();
		cutilCheckError(cutStopTimer(selectPositionTimer));
		logMessage(LOG_LVL_DETAIL,"kernel: getBestFinalMutation2");

		cutilSafeCall(cudaMemcpy(mutationResultBasePos+readIndex, d_bestMutationPos, readBestMutationSize, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(mutationResultValue+readIndex, d_bestMutationIdx, readBestMutationSize, cudaMemcpyDeviceToHost));
		CheckCUDAError();


		for(unsigned int m=0; m<batchSize;m++){
			if(mutationResultBasePos[readIndex+m]!=readLength) mutatedCount++;
		}
        readIndex+=batchSize;
        batchSize=readIndex+maxBatchSize>=readCount?readCount-readIndex:maxBatchSize;
        logMessage(LOG_LVL_DETAIL,"readIndex %d ,batch size %d, readCount %d mutatedCount:%d\n",readIndex, batchSize,readCount,mutatedCount);
        readMutationSize = batchSize* 4 * readLength * readLength * (sizeof(unsigned int));
		readMutationScoreSize = batchSize * 4 * readLength 	* (sizeof(unsigned int));
		readMutationStepSize = batchSize * readLength 	* (sizeof(unsigned int));
		readBestMutationSize= batchSize* (sizeof(unsigned int));
    }

	//mutation/
	*correctedReads = (char *) malloc(
			mutatedCount * readLength * (sizeof(char)));
	unsigned int modifiedReadCount=0;
	for (unsigned int i = 0, mutatedIdx = 0; i < readCount; i++) { //we can move this to cuda
		if (mutationResultBasePos[i] == readLength) {
			//discard it
		} else {
			memcpy(*correctedReads + mutatedIdx * readLength,in_reads + i * readLength, readLength);
			char base=*(*correctedReads + mutatedIdx * readLength+ mutationResultBasePos[i]);
			
			modifiedReadCount+=(mutationResultValue[i] == 0)? 0:1;

			*(*correctedReads + mutatedIdx * readLength+ mutationResultBasePos[i]) = mutator[base & 0x07][mutationResultValue[i]];
			mutatedIdx++;
		}
	}

	setStatItem(TM_MUTATION_GPU_TIME, cutGetTimerValue(mutationTimer));
	setStatItem(TM_ACCUMULATE_GPU_TIME, cutGetTimerValue(accumulateTimer));
	setStatItem(TM_SELECT_MUTATION_GPU_TIME, cutGetTimerValue(selectMutationTimer));
	setStatItem(TM_SELECT_POSITION_GPU_TIME, cutGetTimerValue(selectPositionTimer));

	cutilCheckError(cutDeleteTimer(mutationTimer));
	cutilCheckError(cutDeleteTimer(accumulateTimer));
	cutilCheckError(cutDeleteTimer(selectMutationTimer));
	cutilCheckError(cutDeleteTimer(selectPositionTimer));

	
	correctedReadCount = mutatedCount;
	
	setStatItem(NM_MODIFIED_READ_COUNT,modifiedReadCount);

	deallocateMemory(d_mutationStep);
	deallocateMemory(d_bestMutationPos);
	deallocateMemory(d_bestMutationIdx);
	deallocateMemory(d_buffer);

	deallocateMemory(d_mutationMask);
	deallocateMemory(d_mutationScore);
	deallocateMemory(d_mutation);
	deallocateMemory(d_read);
	deallocateMemory(d_lmerKeys);
	deallocateMemory(d_lmerValues);
	deallocateMemory(d_TK);
	deallocateMemory(d_TV);
	deallocateMemory(d_bucketSize);



	/**/
	free(h_mutation);
	free(h_buffer);

	free(h_mutationStep);
	free(h_bestMutationPos);
	free(h_bestMutationIdx);
	/**/
	free(mutationResultBasePos);
	free(mutationResultValue);
	free(h_mutationScore);
	free(h_mutationMask);
	free(h_lmerKeys);
	free(h_lmerValues);
	return correctedReadCount;
}
