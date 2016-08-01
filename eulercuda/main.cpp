#include <stdio.h>
#include <memory.h>
#include <cutil_inline.h>
#include <math.h>
#include <time.h>       /* defines time_t for timings in the test */
#include <google/sparse_hash_map>
#include <google/dense_hash_map>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/errors.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/version.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/unordered_map.hpp>
#include <boost/foreach.hpp>
#include <iostream>
#include <iterator>
#include "common.h"
#include "Debruijn.h"
#include "Kmer.h"
#include "FastaReader.h"
#include "Graph.h"
#include "utils.h"
#include "component.h"
#include "eulertour.h"
#include "stats.h"
#include "encoder.h"
#include <assert.h>
#include "gpuhash.h"
#include "fixer.h"
#include "path.h"

#include "transform.h"

using namespace boost;
using google::sparse_hash_map;
using google::dense_hash_map;
using namespace std;
namespace po = boost::program_options;
//#define QUICK_DEBUG 1

/**
 GLOBALS
 **/
/*
unsigned char *h_idata;
unsigned int * h_icount;
unsigned int * d_icount;
unsigned char *d_idata;
unsigned long kmerCount;
unsigned long mem_size;
unsigned long kmerFound = 0;
unsigned long readProcessed = 0;*/

/***/
#ifdef EULER_NDEBUG
#define DEBUG_MAIN_CPP(x)
#else
#define DEBUG_MAIN_CPP(x) x
#endif

#define DEBUG_CALL(x) DEBUG_MAIN_CPP(x)



#define GET_EDGE_LABEL(x) (basecode[x & 3])

po::variables_map vm;
char basecode[4]= {'A','C','G','T'};
char translate(int i) {
	if (i == 0)
		return 'A';
	if (i == 1)
		return 'C';
	if (i == 2)
		return 'G';
	if (i == 3)
		return 'T';
	return '.';
}

KEY_T pseudoHash(unsigned char * ch, int count) {

	//had to check if count < log2(KEY_SIZE*8)
	KEY_T value = 0;
	for (int i = 0; i < count; i++) {
		value = value << 2; //mult by 4
		if (*(ch + i) == 'A' || *(ch + i) == 'a') {
			value += 0;
		}
		if (*(ch + i) == 'C' || *(ch + i) == 'c') {
			value += 1;
		}
		if (*(ch + i) == 'G' || *(ch + i) == 'g') {
			value += 2;
		}
		if (*(ch + i) == 'T' || *(ch + i) == 't') {
			value += 3;
		}
	}
	return value;
}
KEY_T pseudoHashComplement(unsigned char * ch, int count) {

	KEY_T value = 0;
	for (int i = count - 1; i >= 0; i--) {
		value = value << 2; //mult by 4
		if (*(ch + i) == 'A' || *(ch + i) == 'a') {
			value += 3;
		}
		if (*(ch + i) == 'C' || *(ch + i) == 'c') {
			value += 2;
		}
		if (*(ch + i) == 'G' || *(ch + i) == 'g') {
			value += 1;
		}
		if (*(ch + i) == 'T' || *(ch + i) == 't') {
			value += 0;
		}
	}
	return value;
}

void getString(char * kmer, int length, KEY_T value) {

	KEY_T currentValue = value;
	for (int i = 1; i <= length; i++) {
		kmer[length - i] = translate((int) (currentValue % 4));
		currentValue = currentValue / 4;
	}
}

void dumpData(unsigned char * data, unsigned int length,
		const char * filename) {
	FILE * file = fopen(filename, "w");
	for (unsigned int i = 0; i < length; i++) {
		fputc(data[i], file);
	}
	fclose(file);

}
void loadData(unsigned char * data, unsigned int length, const char *filename) {
	FILE * file = fopen(filename, "r");
	for (unsigned int i = 0; i < length; i++) {
		data[i] = (unsigned char) fgetc(file);
	}
	fclose(file);
}

void readLmersKmersCuda(	const char * reads,
							unsigned int readLength,
							unsigned int readCount,
							int lmerLength,
							/* output param**/
							KEY_PTR * lmerKeys,
							VALUE_PTR * lmerValues,
							unsigned int * lmerCount,
							KEY_PTR * kmerKeys,
							VALUE_PTR * kmerValues,
							unsigned int * kmerCount,
							PathSystem & P) {


	char * d_reads = NULL;
	KEY_PTR h_lmersF = NULL;
	KEY_PTR h_lmersR = NULL;
	KEY_PTR d_lmers = NULL;
	KEY_PTR h_pkmersF = NULL;
	KEY_PTR h_pkmersR = NULL;
	KEY_PTR h_skmersF = NULL;
	KEY_PTR h_skmersR = NULL;
	KEY_PTR d_pkmers = NULL;
	KEY_PTR d_skmers = NULL;
	unsigned int readProcessed=0;
	unsigned int kmerGPUEncTimer = 0;
	unsigned int kmerExtractTimer=0;

	typedef dense_hash_map<KEY_T, VALUE_T> map;
	map kmerMap(readLength*readCount);
	map lmerMap(readLength*readCount);

	unsigned int lmerEmpty=0;
	unsigned int kmerEmpty=0;

	cutilCheckError(cutCreateTimer(&kmerExtractTimer));
	cutilCheckError(cutStartTimer(kmerExtractTimer));

	kmerMap.set_empty_key((KEY_T)(0xFFFFFFFFFFFFFFFF));
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
	h_pkmersF = (KEY_PTR) malloc(ebSize);
	h_pkmersR = (KEY_PTR) malloc(ebSize);
	h_skmersF = (KEY_PTR) malloc(ebSize);
	h_skmersR = (KEY_PTR) malloc(ebSize);

	//allocate cuda mem
	allocateMemory((void**) &d_reads, bufferSize);
	allocateMemory((void**) &d_lmers, ebSize);
	allocateMemory((void**) &d_skmers, ebSize);
	allocateMemory((void**) &d_pkmers, ebSize);

	KEY_T kmerBitMask = 0;
	for (int j = 0; j < (lmerLength - 1) * 2; j++) {
		kmerBitMask = (kmerBitMask << 1) | 1;
	}

	cutilCheckError(cutCreateTimer(&kmerGPUEncTimer));

	validLmerCount = readLength - lmerLength + 1;
	while(readProcessed<readCount){
		//copy symem to gpumem
		cutilSafeCall(cudaMemcpy(d_reads, (reads+(readProcessed*readLength)), bufferSize,cudaMemcpyHostToDevice));

		cutilCheckError(cutStartTimer(kmerGPUEncTimer));

		//encode
		encodeLmer(d_reads, bufferSize, readLength, d_lmers, lmerLength,entriesCount);

		//extract kmer
		computeKmer(d_lmers, d_pkmers, d_skmers, kmerBitMask, readLength,entriesCount);

		cutilCheckError(cutStopTimer(kmerGPUEncTimer));

		//copy result,
		cutilSafeCall(cudaMemcpy(h_lmersF, d_lmers, ebSize, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(h_pkmersF, d_pkmers, ebSize,cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(h_skmersF, d_skmers, ebSize,cudaMemcpyDeviceToHost));

		cutilCheckError(cutStartTimer(kmerGPUEncTimer));

		//reverse
		//two options 1) read from buffer,2) read from lmer
		//read from buffer is more efficient since 2) would use size(KEY_T)*entriesCount while 1 would use entriesCount.
		encodeLmerComplement(d_reads, bufferSize, readLength, d_lmers,lmerLength, entriesCount);

		//extract kmer
		computeKmer(d_lmers, d_pkmers, d_skmers, kmerBitMask, readLength,entriesCount);

		cutilCheckError(cutStopTimer(kmerGPUEncTimer));

		//copy result
		cutilSafeCall(cudaMemcpy(h_lmersR, d_lmers, ebSize, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(h_pkmersR, d_pkmers, ebSize,cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(h_skmersR, d_skmers, ebSize,cudaMemcpyDeviceToHost));


		//set
		for (unsigned int j = 0; j < readToProcess; j++) {
			for (unsigned int i = 0; i < validLmerCount; i++) {
				unsigned int index = j * readLength + i;

				kmerMap[h_pkmersF[index]] = 1;
				kmerMap[h_skmersF[index]] = 1;
				kmerMap[h_pkmersR[index]] = 1;
				kmerMap[h_skmersR[index]] = 1;

				if (h_lmersF[index] == 0)
					lmerEmpty++;
				else lmerMap[h_lmersF[index]]++;

				if (h_lmersR[index] == 0)
					lmerEmpty++;
				else lmerMap[h_lmersR[index]]++;


				/* for transformation
				string * f;
				string * r;//
				f=new string(1,GET_EDGE_LABEL(h_lmersF[index]));
				r=new string(1,GET_EDGE_LABEL(h_lmersR[index]));
				P.addPathToRead(readProcessed+j,(h_pkmersF[index]),(h_skmersF[index]),*f);
				P.addPathToRead(readCount+readProcessed+j,h_pkmersR[index],h_skmersR[index],*r);

				delete f;
				delete r;

				*/
			}
		}

		readProcessed+=readToProcess;
		readToProcess=readCount-readProcessed;
		readToProcess=readToProcess>CUDA_NUM_READS?CUDA_NUM_READS:readToProcess;	//how many reads to process ,max NUM_CUDA
		entriesCount = readLength *readToProcess; // entries count
		bufferSize = sizeof(char) * readLength * readToProcess; //input buffer Size
		ebSize = entriesCount * sizeof(KEY_T); //entries buffer Size

	}
	cutilCheckError(cutStopTimer(kmerGPUEncTimer));
	setStatItem(TM_KMER_GPUENC_TIME, cutGetTimerValue(kmerGPUEncTimer));
	cutilCheckError(cutDeleteTimer(kmerGPUEncTimer));



	*kmerCount = (unsigned int) kmerMap.size()+kmerEmpty;
	logMessage(LOG_LVL_MSG, "#k-mer count : %d", *kmerCount);
	*kmerKeys = (KEY_PTR) malloc(*kmerCount * KEY_SIZE);
	*kmerValues = (VALUE_PTR) malloc(*kmerCount * VALUE_SIZE);
	unsigned int index = 0;
	BOOST_FOREACH(map::value_type pair, kmerMap) {
		(*kmerKeys)[index]=pair.first;
		(*kmerValues)[index]=index;
		index++;
	}
	if(kmerEmpty>0){
		(*kmerKeys)[index]=0;
		(*kmerValues)[index]=index;
	}

	*lmerCount = (unsigned int) lmerMap.size()+(lmerEmpty>0?1:0);
	logMessage(LOG_LVL_MSG, "#l-mer count : %d", *lmerCount);
	*lmerKeys = (KEY_PTR) malloc(*lmerCount * KEY_SIZE);
	*lmerValues = (VALUE_PTR) malloc(*lmerCount * VALUE_SIZE);

	index = 0;
	BOOST_FOREACH(map::value_type pair, lmerMap) {
		(*lmerKeys)[index]=pair.first;
		(*lmerValues)[index]=pair.second;
		index++;
	}
	if(lmerEmpty>0){
			(*lmerKeys)[index]=0;
			(*lmerValues)[index]=lmerEmpty;
		}
	deallocateMemory(d_reads);
	deallocateMemory(d_lmers);
	deallocateMemory(d_pkmers);
	deallocateMemory(d_skmers);



	free(h_lmersF);
	free(h_lmersR);
	free(h_pkmersF);
	free(h_pkmersR);
	free(h_skmersF);
	free(h_skmersR);

	cutilCheckError(cutStopTimer(kmerExtractTimer));
	setStatItem(TM_KMER_EXTRACTION_TIME, cutGetTimerValue(kmerExtractTimer));
	cutilCheckError(cutDeleteTimer(kmerExtractTimer));

	//TODO reset device
}
void readLmersKmersg(const char * path, int lmerLength, KEY_PTR * lmerKeys,
		VALUE_PTR * lmerValues, unsigned int * lmerCount, KEY_PTR * kmerKeys,
		VALUE_PTR * kmerValues, unsigned int * kmerCount) {
	KmerReader * kmerReader = createKmerReader(path, lmerLength);
	KEY_T key;
	KEY_T prefixKey;
	KEY_T suffixKey;
	typedef dense_hash_map<KEY_T, VALUE_T> map;
	map kmerMap;
	map lmerMap;

	kmerMap.set_empty_key(NULL);
	lmerMap.set_empty_key(NULL);
	getNext(kmerReader);
	while (kmerReader->status != -1) {
		key = pseudoHash(kmerReader->kmer, lmerLength);
		prefixKey = pseudoHash(kmerReader->kmer, lmerLength - 1);
		suffixKey = pseudoHash(kmerReader->kmer + 1, lmerLength - 1);
		kmerMap[prefixKey] = 1;
		kmerMap[suffixKey] = 1;
		lmerMap[key]++;
		key = pseudoHashComplement(kmerReader->kmer, lmerLength);
		prefixKey = pseudoHashComplement(kmerReader->kmer, lmerLength - 1);
		suffixKey = pseudoHashComplement(kmerReader->kmer + 1, lmerLength - 1);
		kmerMap[prefixKey] = 1;
		kmerMap[suffixKey] = 1;
		lmerMap[key]++;
		getNext(kmerReader);
	}

	destroyKmerReader(kmerReader);
	*kmerCount = (unsigned int) kmerMap.size();
	logMessage(LOG_LVL_MSG, "#k-mer count : %d", *kmerCount);
	*kmerKeys = (KEY_PTR) malloc(*kmerCount * KEY_SIZE);
	*kmerValues = (VALUE_PTR) malloc(*kmerCount * VALUE_SIZE);
	unsigned int index = 0;
	BOOST_FOREACH(map::value_type pair, kmerMap) {
		(*kmerKeys)[index]=pair.first;
		(*kmerValues)[index]=index;
		index++;
	}
	*lmerCount = (unsigned int) lmerMap.size();
	logMessage(LOG_LVL_MSG, "#l-mer count : %d", *lmerCount);
	*lmerKeys = (KEY_PTR) malloc(*lmerCount * KEY_SIZE);
	*lmerValues = (VALUE_PTR) malloc(*lmerCount * VALUE_SIZE);

	index = 0;
BOOST_FOREACH(map::value_type pair, lmerMap) {
	(*lmerKeys)[index]=pair.first;
	(*lmerValues)[index]=pair.second;
	index++;
}

}
void generatePartialContigHost(const char * outputFileName, EulerVertex * h_ev,
		unsigned int vcount, EulerEdge * h_ee, unsigned int ecount,
		int kmerLength) {

	unsigned char * d_contigStart;
	unsigned char * h_contigStart;
	unsigned char * h_visited;
	EulerEdge * d_ee;
	char * buffer;

	FILE * ofile;
	allocateMemory((void**) &d_contigStart, sizeof(unsigned char) * ecount);
	h_contigStart = (unsigned char *) malloc(ecount);

	allocateMemory((void**) &d_ee, sizeof(EulerEdge) * ecount);
	cutilSafeCall(
			cudaMemcpy(d_ee, h_ee, sizeof(EulerEdge) * ecount,
					cudaMemcpyHostToDevice));
	markContigStart(d_ee, d_contigStart, ecount);

	h_visited = (unsigned char *) malloc(sizeof(unsigned char) * ecount);
	buffer = (char *) malloc(kmerLength);
	memset(buffer, 0, kmerLength);
	memset(h_visited, 0, ecount);

	//change file name
	ofile = fopen(outputFileName, "w");

	unsigned int count = 0;
	unsigned int edgeCount = 0;
	unsigned int next;
	for (unsigned int i = 0; i < ecount; i++) {
		if (h_contigStart[i] != 0 && h_visited[i] == 0) {

			fprintf(ofile, ">%u\n", count);
			logMessage(LOG_LVL_INFO, ">%lu\n", count);
			count++;
			getString(buffer, kmerLength - 1, h_ev[h_ee[i].v1].vid);
			fprintf(ofile, "%s", buffer);
			logMessageNL(LOG_LVL_INFO, "%s", buffer);
			//h_visited[i]=1;

			next = i;
			while (h_ee[next].s < ecount && h_visited[h_ee[next].s] == 0) {
				h_visited[next] = 1;
				next = h_ee[next].s;
				getString(buffer, kmerLength - 1, h_ev[h_ee[next].v1].vid);
				fprintf(ofile, "%s", buffer + kmerLength - 2);
				logMessageNL(LOG_LVL_INFO, "%s", buffer + kmerLength - 2);

				edgeCount++;
			}
			if (h_visited[next] == 0) { // for circular paths
				getString(buffer, kmerLength - 1, h_ev[h_ee[next].v2].vid);
				fprintf(ofile, "%s", buffer + kmerLength - 2);
				logMessageNL(LOG_LVL_INFO, "%s", buffer + kmerLength - 2);
				h_visited[next] = 1;
				edgeCount++;
			}
			fprintf(ofile, "\n\n");
			logMessage(LOG_LVL_INFO, "\n");
		}

	}
//printf("generated %u contigs from source vertices\nstarting  circular\n",count);
	for (unsigned int i = 0; i < ecount; i++) {
		if (h_visited[i] == 0) {

			fprintf(ofile, ">%u\n", count);
			logMessage(LOG_LVL_INFO, ">%lu", count);
			count++;
			getString(buffer, kmerLength - 1, h_ev[h_ee[i].v1].vid);
			fprintf(ofile, "%s", buffer);
			logMessageNL(LOG_LVL_INFO, "%s", buffer);

			next = i;
			while (h_ee[next].s < ecount && h_visited[h_ee[next].s] == 0) {
				h_visited[next] = 1;
				next = h_ee[next].s;
				getString(buffer, kmerLength - 1, h_ev[h_ee[next].v1].vid);
				fprintf(ofile, "%s", buffer + kmerLength - 2);
				logMessageNL(LOG_LVL_INFO, "%s", buffer + kmerLength - 2);

				edgeCount++;
			}
			if (h_visited[next] == 0) { // for circular paths
				getString(buffer, kmerLength - 1, h_ev[h_ee[next].v2].vid);
				fprintf(ofile, "%s", buffer + kmerLength - 2);
				logMessageNL(LOG_LVL_INFO, "%s", buffer + kmerLength - 2);
				h_visited[next] = 1;
				edgeCount++;
			}
			fprintf(ofile, "\n\n");
			logMessage(LOG_LVL_INFO, "\n");
		}

	}

	logMessage(
			LOG_LVL_MSG,
			"GOLD - Total contig generated : %u\n#edges consumed : %u\n#total edges : %u",
			count, edgeCount, ecount);
	fclose(ofile);
	free(h_contigStart);
	free(h_visited);
	free(buffer);
	deallocateMemory(d_contigStart);
	deallocateMemory(d_ee);
}

void generatePartialContig(const char * outputFileName, EulerVertex * d_ev,
		unsigned int vcount, EulerEdge * d_ee, unsigned int ecount,
		int kmerLength) {/*kmerLenght is actually lmerLength*/

	unsigned char * d_contigStart;
	unsigned char * h_contigStart;
	unsigned char * h_visited;
	EulerEdge * h_ee;
	EulerVertex * h_ev;
	char * buffer;

	FILE * ofile;
	allocateMemory((void**) &d_contigStart, sizeof(unsigned char) * ecount);
	h_contigStart = (unsigned char *) malloc(ecount);
	markContigStart(d_ee, d_contigStart, ecount);

	cutilSafeCall(
			cudaMemcpy(h_contigStart, d_contigStart, ecount,
					cudaMemcpyDeviceToHost));

	h_ev = (EulerVertex *) malloc(sizeof(EulerVertex) * vcount);
	h_ee = (EulerEdge *) malloc(sizeof(EulerEdge) * ecount);
	h_visited = (unsigned char *) malloc(sizeof(unsigned char) * ecount);
	buffer = (char *) malloc(kmerLength);
	memset(buffer, 0, kmerLength);
	memset(h_visited, 0, ecount);

	cutilSafeCall(
			cudaMemcpy(h_ev, d_ev, sizeof(EulerVertex) * vcount,
					cudaMemcpyDeviceToHost));
	cutilSafeCall(
			cudaMemcpy(h_ee, d_ee, sizeof(EulerEdge) * ecount,
					cudaMemcpyDeviceToHost));

	ofile = fopen(outputFileName, "w");

	unsigned int count = 0;
	unsigned int edgeCount = 0;
	unsigned int next;
	for (unsigned int i = 0; i < ecount; i++) {
		if (h_contigStart[i] != 0 && h_visited[i] == 0) {

			fprintf(ofile, ">%u\n", count);
			logMessage(LOG_LVL_ALL, ">%lu\n", count);
			count++;
			getString(buffer, kmerLength - 1, h_ev[h_ee[i].v1].vid);
			fprintf(ofile, "%s", buffer);
			logMessageNL(LOG_LVL_ALL, "%s", buffer);
			//h_visited[i]=1;

			next = i;
			while (h_ee[next].s < ecount && h_visited[h_ee[next].s] == 0) {
				h_visited[next] = 1;
				next = h_ee[next].s;
				getString(buffer, kmerLength - 1, h_ev[h_ee[next].v1].vid);
				fprintf(ofile, "%s", buffer + kmerLength - 2);
				logMessageNL(LOG_LVL_ALL, "%s", buffer + kmerLength - 2);

				edgeCount++;
			}
			if (h_visited[next] == 0) { // for circular paths
				getString(buffer, kmerLength - 1, h_ev[h_ee[next].v2].vid);
				fprintf(ofile, "%s", buffer + kmerLength - 2);
				logMessageNL(LOG_LVL_ALL, "%s", buffer + kmerLength - 2);
				h_visited[next] = 1;
				edgeCount++;
			}
			fprintf(ofile, "\n\n");
			logMessage(LOG_LVL_ALL, "\n");
		}

	}
//printf("generated %u contigs from source vertices\nstarting  circular\n",count);
	for (unsigned int i = 0; i < ecount; i++) {
		if (h_visited[i] == 0) {

			fprintf(ofile, ">%u\n", count);
			logMessage(LOG_LVL_ALL, ">%lu", count);
			count++;
			getString(buffer, kmerLength - 1, h_ev[h_ee[i].v1].vid);
			fprintf(ofile, "%s", buffer);
			logMessageNL(LOG_LVL_ALL, "%s", buffer);

			next = i;
			while (h_ee[next].s < ecount && h_visited[h_ee[next].s] == 0) {
				h_visited[next] = 1;
				next = h_ee[next].s;
				getString(buffer, kmerLength - 1, h_ev[h_ee[next].v1].vid);
				fprintf(ofile, "%s", buffer + kmerLength - 2);
				logMessageNL(LOG_LVL_ALL, "%s", buffer + kmerLength - 2);

				edgeCount++;
			}
			if (h_visited[next] == 0) { // for circular paths
				getString(buffer, kmerLength - 1, h_ev[h_ee[next].v2].vid);
				fprintf(ofile, "%s", buffer + kmerLength - 2);
				logMessageNL(LOG_LVL_ALL, "%s", buffer + kmerLength - 2);
				h_visited[next] = 1;
				edgeCount++;
			}
			fprintf(ofile, "\n\n");
			logMessage(LOG_LVL_ALL, "\n");
		}

	}

	logMessage(
			LOG_LVL_MSG,
			"Total contig generated : %u\n#edges consumed : %u\n#total edges : %u",
			count, edgeCount, ecount);
	fclose(ofile);

	free(h_ev);
	free(h_ee);
	free(h_visited);
	free(buffer);
	free(h_contigStart);
	deallocateMemory(d_contigStart);
}

void printKmer(unsigned char * kmers, unsigned int length) {
	unsigned int index = 0;
	for (unsigned int i = 0; i < length; i++) {
		unsigned char b = 1;
		for (unsigned char j = 0; j < 8; j++) {
			printf("[%u]: %d\n", index, (kmers[i] & b));
			b = b << 1;
			index++;
		}
	}
}
void findConflicts(EulerVertex * d_ev, unsigned int vcount) {
	EulerVertex * h_ev;
	h_ev = (EulerVertex *) malloc(sizeof(EulerVertex) * vcount);
	cutilSafeCall(
			cudaMemcpy(h_ev, d_ev, sizeof(EulerVertex) * vcount,
					cudaMemcpyDeviceToHost));

	unsigned int problemVertexCount = 0;
	unsigned int mismatch = 0;
	unsigned int moretochoose = 0;
	for (unsigned int i = 0; i < vcount; i++) {

		if (h_ev[i].ecount != h_ev[i].lcount && h_ev[i].ecount > 0
				&& h_ev[i].lcount > 0) {
			mismatch++;
			problemVertexCount++;
		} else if (h_ev[i].ecount > 1 || h_ev[i].lcount > 1) {
			moretochoose++;
			problemVertexCount++;
		}
	}
	printf(
			"#Total conflicting vertices: %u  \n#DegreeMisMatch : %u \n#MorethanOneDegree : %u \n",
			problemVertexCount, mismatch, moretochoose);
	free(h_ev);
}

void assembleGold(const char * fileName, const char * output, unsigned int l,
		unsigned int coverage) {

	EulerVertex * h_ev;
	unsigned int * h_l;
	unsigned int * h_e;
	EulerEdge * h_ee;
	unsigned int ecount;
	KEY_PTR h_kmerKeys;
	VALUE_PTR h_kmerValues;
	KEY_PTR h_lmerKeys;
	VALUE_PTR h_lmerValues;
	KEY_PTR d_kmerKeys;
	VALUE_PTR d_kmerValues;
	KEY_PTR d_lmerKeys;
	VALUE_PTR d_lmerValues;
	unsigned int bucketCount;
	unsigned int tableLength;
	KEY_PTR d_TK;
	KEY_PTR h_TK;

	VALUE_PTR d_TV;
	VALUE_PTR h_TV;
	unsigned int * d_bucketSeed;
	unsigned int * h_bucketSeed;
	unsigned int lmerCount;
	unsigned int kmerCount;

//	readLmersKmersCuda(fileName,l,&h_lmerKeys,&h_lmerValues,&lmerCount,&h_kmerKeys,&h_kmerValues,&kmerCount,256);
//	readLmersKmers(fileName, l, &h_lmerKeys, &h_lmerValues, &lmerCount,
	//		&h_kmerKeys, &h_kmerValues, &kmerCount);
	for (unsigned int i = 0; i < lmerCount; i++) {
		double value;
		if (h_lmerValues[i] < coverage) {
			if (h_lmerValues[i] > 0) {
				h_lmerValues[i] = 1;
			}
		} else {
			value = (double) (h_lmerValues[i]) / (double) (coverage);
			h_lmerValues[i] = (unsigned int) (value + 0.5);
		}
	}

	allocateMemory((void**) &d_lmerKeys, lmerCount * (KEY_SIZE));
	allocateMemory((void**) &d_lmerValues, lmerCount * (VALUE_SIZE));
	allocateMemory((void**) &d_kmerKeys, kmerCount * (KEY_SIZE));
	allocateMemory((void**) &d_kmerValues, kmerCount * (VALUE_SIZE));

	cutilSafeCall(
			cudaMemcpy(d_lmerKeys, h_lmerKeys, lmerCount * (KEY_SIZE),
					cudaMemcpyHostToDevice));
	cutilSafeCall(
			cudaMemcpy(d_lmerValues, h_lmerValues, lmerCount * (VALUE_SIZE),
					cudaMemcpyHostToDevice));
	cutilSafeCall(
			cudaMemcpy(d_kmerKeys, h_kmerKeys, kmerCount * (KEY_SIZE),
					cudaMemcpyHostToDevice));
	cutilSafeCall(
			cudaMemcpy(d_kmerValues, h_kmerValues, kmerCount * (VALUE_SIZE),
					cudaMemcpyHostToDevice));

	createHashTable(d_kmerKeys, d_kmerValues, kmerCount, &d_TK, &d_TV,
			&tableLength, &d_bucketSeed, &bucketCount);
	//copy h_TK, h_TV, h_bucketSeed
	h_TK = (KEY_PTR) malloc(bucketCount * BUCKET_KEY_SIZE);
	h_TV = (VALUE_PTR) malloc(bucketCount * BUCKET_VALUE_SIZE);
	h_bucketSeed = (unsigned int *) malloc(bucketCount * sizeof(unsigned int));

	cutilSafeCall(
			cudaMemcpy(h_TK, d_TK, bucketCount * (BUCKET_KEY_SIZE),
					cudaMemcpyDeviceToHost));
	cutilSafeCall(
			cudaMemcpy(h_TV, d_TV, bucketCount * (BUCKET_VALUE_SIZE),
					cudaMemcpyDeviceToHost));
	cutilSafeCall(
			cudaMemcpy(h_bucketSeed, d_bucketSeed,
					bucketCount * (sizeof(unsigned int)),
					cudaMemcpyDeviceToHost));
	constructDebruijnGraphGold(h_lmerKeys, h_lmerValues, lmerCount, h_kmerKeys,
			kmerCount, l, h_TK, h_TV, h_bucketSeed, bucketCount, &h_ev, &h_l,
			&h_e, &h_ee, &ecount);
//		setStatItem(NM_DEBRUIJN_VTX,kmerCount);
//		setStatItem(NM_DEBRUIJN_EDG,ecount);

	deallocateMemory(d_lmerKeys);
	deallocateMemory(d_lmerValues);
	deallocateMemory(d_kmerKeys);
	deallocateMemory(d_kmerValues);
	deallocateMemory(d_TK);
	deallocateMemory(d_TV);
	deallocateMemory(d_bucketSeed);
	free(h_lmerKeys);
	free(h_lmerValues);
	free(h_kmerKeys);
	free(h_kmerValues);
	free(h_TK);
	free(h_TV);
	free(h_bucketSeed);

//	findEulerGold(h_ev, h_l, h_e, kmerCount, h_ee, ecount, l);

	generatePartialContigHost(output, h_ev, kmerCount, h_ee, ecount, l);

	free(h_ev);
	free(h_l);
	free(h_e);
	free(h_ee);
}


/*processReads-- read input file and store reads in readBuffer
 * read buffer will be accomodated to number of reads processed
 * */
unsigned int processReads(	const char * filename,		//input file
					unsigned int readLength,	//read length
					char **	readBuffer			//out put buffer
					){

	unsigned int readCount=0;
	char * tempBuff=NULL;
	FILE * fptr=NULL;
	//allocate very big buffer and truncate it later on, we might use linked lst


	unsigned int expectedReadCount=4*1024*1024;// 4 million reads
	unsigned int maxBufferSize=sizeof(char)*readLength*expectedReadCount;

	unsigned int processeReadsTimer=0;

	cutilCheckError(cutCreateTimer(&processeReadsTimer));
	cutilCheckError(cutStartTimer(processeReadsTimer));


	tempBuff=(char *)malloc(maxBufferSize);

	fptr=fopen(filename,"r");
	if(fptr!=NULL){
		readCount=getReads(fptr,tempBuff,maxBufferSize,readLength,expectedReadCount);
		if(expectedReadCount>readCount){
			unsigned int newBufferSize=readCount*readLength;
			*readBuffer=(char *)malloc(newBufferSize);
			memcpy(*readBuffer,tempBuff,newBufferSize);
		}else{
			//need larger buffer,
			*readBuffer=NULL;
		}
	}else{
		//log error opening file
	}

	cutilCheckError(cutStopTimer(processeReadsTimer));
	setStatItem(TM_PROCESS_READ_TIME, cutGetTimerValue(processeReadsTimer));
	setStatItem(NM_READ_COUNT, readCount);
	cutilCheckError(cutDeleteTimer(processeReadsTimer));


	if(tempBuff!=NULL) free(tempBuff);
	if(fptr!=NULL) fclose(fptr);
	return readCount;
}
void writeCorrectedReads(char * correctedReads, unsigned int correctedReadCount, unsigned int readLength){

	FILE * fptr=fopen(vm["output-file"].as<string>().c_str(),"w");
	char * buffer=NULL;
	buffer=(char *)malloc(readLength+1);
	buffer[readLength]='\0';
	for(int i=0;i<correctedReadCount; i++){
		memcpy(buffer,correctedReads+i*readLength,readLength);
		fprintf(fptr,">%i\n%s\n",i,buffer);
	}
	free(buffer);
	fclose(fptr);
}
/* doErrorCorrection, fix input reads
 * */
unsigned int doErrorCorrection(	char ** readBuffer,		//input readBuffer
								unsigned int readCount,	//num reads
								unsigned int readLength,	//read length
								unsigned int tuple_size,	//l
								unsigned int positions
								){
	unsigned int correctedReadCount=readCount;
	char * correctedReads=NULL;
	unsigned int errorCorrectionTimer=0;

	cutilCheckError(cutCreateTimer(&errorCorrectionTimer));
	cutilCheckError(cutStartTimer(errorCorrectionTimer));


	for(unsigned int i=0;i<positions;i++){
		correctedReadCount=errorCorrection(*readBuffer,correctedReadCount,readLength,&correctedReads,tuple_size);
		assert(correctedReads!=NULL);
		free(*readBuffer);
		*readBuffer=correctedReads;
	}


	cutilCheckError(cutStopTimer(errorCorrectionTimer));
	setStatItem(TM_ERROR_CORRECTION_TIME, cutGetTimerValue(errorCorrectionTimer));
	setStatItem(NM_CORRECTED_READ_COUNT, correctedReadCount);
	cutilCheckError(cutDeleteTimer(errorCorrectionTimer));

	///DEBUG_CALL(writeCorrectedReads(correctedReads,correctedReadCount,readLength));
	if(!vm.count("assemble"))		{	//write if not assemble
		(writeCorrectedReads(correctedReads,correctedReadCount,readLength));
	}

	return correctedReadCount;
}
/*
void checkHash(KEY_PTR h_lmerKeys,KEY_PTR d_TK, VALUE_PTR d_TV,unsigned int tableLength,unsigned int * d_bucketSize , unsigned int bucketCount){

	KEY_PTR h_TK=NULL;
	KEY_PTR h_TV=NULL;
	unsigned int * h_bucketSize=NULL;

	h_TK=(KEY_PTR) malloc();
	h_TV=(VALUE_PTR)malloc();
	h_bucketSize=(unsigned int *) malloc(bucketCount*sizeof(unsigned int ));
	cutilSafeCall(
				cudaMemcpy(h_TK, d_TK, lmerCount * (KEY_SIZE),
						cudaMemcpyDeviceToDevice));


	free(h_TK);
	free(h_TV);
	free(h_bucketSize);
}*/

unsigned int constructDebruijnGraph(char * readBuffer,
									unsigned int readCount,
									unsigned int readLength,
									unsigned int l,
									/*output params*/
									EulerVertex ** ev,
									EulerEdge 	** ee,
									unsigned int ** levEdge,
									unsigned int ** entEdge,
									unsigned int * edgeCount,
									unsigned int * vertexCount,
									PathSystem & P){


	///variables
	KEY_PTR			h_lmerKeys =NULL;
	VALUE_PTR 		h_lmerValues= NULL;
	KEY_PTR			d_lmerKeys =NULL;
	VALUE_PTR 		d_lmerValues= NULL;
	unsigned int 	lmerCount=0;
	KEY_PTR 		h_kmerKeys=NULL;
	VALUE_PTR 		h_kmerValues=NULL;
	KEY_PTR 		d_kmerKeys=NULL;
	VALUE_PTR 		d_kmerValues=NULL;
	unsigned int 	kmerCount=0;
	KEY_PTR			d_TK=NULL;
	VALUE_PTR		d_TV=NULL;
	unsigned int 	tableLength=0;
	unsigned int	bucketCount=0;
	unsigned int *	d_bucketSize=NULL;

	unsigned int coverage =20;

	EulerVertex * d_ev=NULL;
	EulerEdge 	* d_ee=NULL;
	unsigned int * d_levEdge=NULL;
	unsigned int * d_entEdge=NULL;



	//timers

	unsigned int hashTableTimer=0;
	unsigned int debruijnTimer = 0;
	// initdevice
	initDevice();

	resetDevice();//does nothing


	//extract kmerLmers
	readLmersKmersCuda(readBuffer,readLength,readCount, l, &h_lmerKeys, &h_lmerValues, &lmerCount,&h_kmerKeys, &h_kmerValues, &kmerCount,P);

	initDevice();

	setStatItem(NM_LMER_COUNT, lmerCount);
	setStatItem(NM_KMER_COUNT, kmerCount);
	for (unsigned int i = 0; i < lmerCount; i++) {
		double value;
		if (h_lmerValues[i] < coverage) {
			if (h_lmerValues[i] > 0) {
				h_lmerValues[i] = 1;
			}
		} else {
			value = (double) (h_lmerValues[i]) / (double) (coverage);
			h_lmerValues[i] = (unsigned int) (value + 0.5);
		}
	}

	allocateMemory((void**) &d_lmerKeys, lmerCount * (KEY_SIZE));
	allocateMemory((void**) &d_lmerValues, lmerCount * (VALUE_SIZE));
	allocateMemory((void**) &d_kmerKeys, kmerCount * (KEY_SIZE));
	allocateMemory((void**) &d_kmerValues, kmerCount * (VALUE_SIZE));


	cutilCheckError(cutCreateTimer(&hashTableTimer));
	cutilCheckError(cutStartTimer(hashTableTimer));
	//printKmer(h_idata,mem_size);
	cutilSafeCall(
			cudaMemcpy(d_lmerKeys, h_lmerKeys, lmerCount * (KEY_SIZE),
					cudaMemcpyHostToDevice));
	cutilSafeCall(
			cudaMemcpy(d_lmerValues, h_lmerValues, lmerCount * (VALUE_SIZE),
					cudaMemcpyHostToDevice));
	cutilSafeCall(
			cudaMemcpy(d_kmerKeys, h_kmerKeys, kmerCount * (KEY_SIZE),
					cudaMemcpyHostToDevice));
	cutilSafeCall(
			cudaMemcpy(d_kmerValues, h_kmerValues, kmerCount * (VALUE_SIZE),
					cudaMemcpyHostToDevice));

	createHashTable(d_kmerKeys, d_kmerValues, kmerCount, &d_TK, &d_TV,
			&tableLength, &d_bucketSize, &bucketCount);
	cutilCheckError(cutStopTimer(hashTableTimer));
	setStatItem(TM_HASHTABLE_CONSTRUCTION, cutGetTimerValue(hashTableTimer));
	cutilCheckError(cutDeleteTimer(hashTableTimer));

	//checkHash(h_lmerKeys,d_TK,d_TV,tableLength,d_bucketSize,bucketCount);


	cutilCheckError(cutCreateTimer(&debruijnTimer));
	cutilCheckError(cutStartTimer(debruijnTimer));

	constructDebruijnGraphDevice(d_lmerKeys, d_lmerValues, lmerCount,
			d_kmerKeys, kmerCount, l, d_TK, d_TV, d_bucketSize, bucketCount,
			&d_ev, &d_levEdge, &d_entEdge, &d_ee, edgeCount);
	*vertexCount=kmerCount;
	cutilCheckError(cutStopTimer(debruijnTimer));
	setStatItem(TM_DEBRUIJN_CONSTRUCTION, cutGetTimerValue(debruijnTimer));
	setStatItem(NM_DEBRUIJN_VTX, kmerCount);
	setStatItem(NM_DEBRUIJN_EDG, *edgeCount);
	cutilCheckError(cutDeleteTimer(debruijnTimer));
//	}
	//copy graph
	*ev=(EulerVertex *)malloc(sizeof(EulerVertex)* (*vertexCount));
	*ee=(EulerEdge *)malloc(sizeof(EulerEdge)* (*edgeCount));
	*levEdge=(unsigned int *)malloc(sizeof(unsigned int)* (*edgeCount));
	*entEdge=(unsigned int * )malloc(sizeof(unsigned int)*(*edgeCount));

	cutilSafeCall(
				cudaMemcpy(*ev, d_ev, sizeof(EulerVertex)*(*vertexCount),
						cudaMemcpyDeviceToHost));
	cutilSafeCall(
				cudaMemcpy(*ee, d_ee, sizeof(EulerEdge)*(*edgeCount),
						cudaMemcpyDeviceToHost));
	cutilSafeCall(
				cudaMemcpy(*levEdge, d_levEdge, sizeof(unsigned int)*(*edgeCount),
						cudaMemcpyDeviceToHost));
	cutilSafeCall(
				cudaMemcpy(*entEdge, d_entEdge, sizeof(unsigned int)*(*edgeCount),
						cudaMemcpyDeviceToHost));

	//free mem
	deallocateMemory(d_lmerKeys);
	deallocateMemory(d_lmerValues);
	deallocateMemory(d_kmerKeys);
	deallocateMemory(d_kmerValues);
	deallocateMemory(d_TK);
	deallocateMemory(d_TV);
	deallocateMemory(d_bucketSize);
	deallocateMemory(d_ev);
	deallocateMemory(d_ee);
	deallocateMemory(d_levEdge);
	deallocateMemory(d_entEdge);
	free(h_lmerKeys);
	free(h_lmerValues);
	free(h_kmerKeys);
	free(h_kmerValues);


	// reset device
	resetDevice();
	return 0;
}

unsigned int findSpanningTree(	CircuitEdge * cg_edge,
								unsigned int cg_edgecount,
								unsigned int cg_vertexcount,
								/**out param*/
								unsigned int ** tree

		){

	int * weights=NULL;
	typedef adjacency_list<listS, vecS, undirectedS, no_property,
		property<edge_index_t, int, property<edge_weight_t, int> > > Graph;
	typedef graph_traits<Graph>::edge_descriptor Edge;
	typedef graph_traits<Graph>::vertex_descriptor Vertex;

	weights = (int *) malloc(sizeof(int) * cg_edgecount);
	for (unsigned int i = 0; i < cg_edgecount; i++) {
	weights[i] = 1;
	}
	Graph g(cg_vertexcount);
	property_map<Graph, edge_weight_t>::type weightmap = get(edge_weight,
		g);
	property_map<Graph, edge_index_t>::type indexmap = get(edge_index, g);
	for (std::size_t j = 0; j < cg_edgecount; ++j) {
	Edge e;
	bool inserted;
	tie(e, inserted) = add_edge(cg_edge[j].c1, cg_edge[j].c2, g);
	weightmap[e] = weights[j];
	indexmap[e] = (int) j;
	}

	property_map<Graph, edge_index_t>::type index = get(edge_index, g);
	std::vector<Edge> spanning_tree;
	kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));

	*tree = (unsigned int *) malloc(
		spanning_tree.size() * sizeof(unsigned int));
	memset(*tree, 0, spanning_tree.size() * sizeof(unsigned int));
	unsigned i = 0;
	logMessage(LOG_LVL_DEBUG, "$$Tree Edges");
	for (std::vector<Edge>::iterator ei = spanning_tree.begin();
		ei != spanning_tree.end(); ++ei) {
	(*tree)[i] = index[*ei];
	//logMessage(LOG_LVL_DEBUG,"$edge index %u ",tree[i]);
	i++;
	}

	free(weights);
	return spanning_tree.size() ;
}
unsigned int findEulerTour(	EulerVertex * ev,
							EulerEdge 	* ee,
							unsigned int * levEdge,
							unsigned int * entEdge,
							unsigned int  edgeCount,
							unsigned int  vertexCount,
							unsigned int l,
							//output params
							const char * outputFileName){



	/*variables*/
	EulerVertex * d_ev=NULL;
	EulerEdge 	* d_ee=NULL;
	unsigned int * d_levEdge=NULL;
	unsigned int * d_entEdge=NULL;

	CircuitEdge * d_cg_edge=NULL;
	unsigned int cg_edgecount=0;
	unsigned int cg_vertexcount=0;
	unsigned int * tree=NULL;
	unsigned int * d_tree;
	unsigned int treeSize=0;
	/* Timers*/
	unsigned int eulerTimer = 0;
	unsigned int mstTimer = 0;
	unsigned int swipeTimer = 0;
	unsigned int partialContigTimer = 0;
	//init device
	initDevice();

	cutilCheckError(cutCreateTimer(&eulerTimer));
	cutilCheckError(cutStartTimer(eulerTimer));

	// copy graph to device
	allocateMemory((void**) &d_ev, sizeof(EulerVertex) * (vertexCount));
		allocateMemory((void**) &d_levEdge, sizeof(unsigned int) * (edgeCount));
		allocateMemory((void**) &d_entEdge, sizeof(unsigned int) * (edgeCount));
		allocateMemory((void**) &d_ee, sizeof(EulerEdge) * (edgeCount));
	cutilSafeCall(
				cudaMemcpy(d_ev, ev, sizeof(EulerVertex)*(vertexCount),
						cudaMemcpyHostToDevice));
	cutilSafeCall(
				cudaMemcpy(d_ee, ee, sizeof(EulerEdge)*(edgeCount),
						cudaMemcpyHostToDevice));
	cutilSafeCall(
				cudaMemcpy(d_levEdge, levEdge, sizeof(unsigned int)*(edgeCount),
						cudaMemcpyHostToDevice));
	cutilSafeCall(
				cudaMemcpy(d_entEdge, entEdge, sizeof(unsigned int)*(edgeCount),
						cudaMemcpyHostToDevice));
	//execute euler
	findEulerDevice(d_ev, d_levEdge, d_entEdge, vertexCount, d_ee, edgeCount, &d_cg_edge,
			&cg_edgecount, &cg_vertexcount, l);
	setStatItem(NM_CIRCUIT_VTX, cg_vertexcount);
	setStatItem(NM_CIRCUIT_EDG, cg_edgecount);
	if (cg_edgecount > 0) { //if cg has some edges


		cutilCheckError(cutCreateTimer(&mstTimer));
		cutilCheckError(cutStartTimer(mstTimer));

		CircuitEdge * cg_edge = (CircuitEdge *) (malloc(
				sizeof(CircuitEdge) * cg_edgecount));
		cutilSafeCall(
				cudaMemcpy(cg_edge, d_cg_edge,
						sizeof(CircuitEdge) * cg_edgecount,
						cudaMemcpyDeviceToHost));


		treeSize=findSpanningTree(cg_edge,cg_edgecount,cg_vertexcount,&tree);
		allocateMemory((void**) &d_tree,
				treeSize * sizeof(unsigned int));
		cutilSafeCall(
				cudaMemcpy(d_tree, tree,
						treeSize * sizeof(unsigned int),
						cudaMemcpyHostToDevice));
		cutilCheckError(cutStopTimer(mstTimer));
		setStatItem(TM_SPANNING_TREE, cutGetTimerValue(mstTimer));
		cutilCheckError(cutDeleteTimer(mstTimer));


		cutilCheckError(cutCreateTimer(&swipeTimer));
		cutilCheckError(cutStartTimer(swipeTimer));
		executeSwipeDevice(d_ev, d_entEdge, vertexCount, d_ee, edgeCount, d_cg_edge,
				cg_edgecount, d_tree, treeSize);
		cutilCheckError(cutStopTimer(swipeTimer));
		setStatItem(TM_SWIPE_EXECUTION, cutGetTimerValue(swipeTimer));
		cutilCheckError(cutDeleteTimer(swipeTimer));

		deallocateMemory(d_tree);
		deallocateMemory(d_cg_edge);
		free(cg_edge);
		free(tree);


	}

	cutilCheckError(cutStopTimer(eulerTimer));
	setStatItem(TM_EULER_TOUR, cutGetTimerValue(eulerTimer));
	cutilCheckError(cutDeleteTimer(eulerTimer));
	/*generate contig*/


	cutilCheckError(cutCreateTimer(&partialContigTimer));
	cutilCheckError(cutStartTimer(partialContigTimer));
	generatePartialContig(outputFileName, d_ev, vertexCount, d_ee, edgeCount, l);
	cutilCheckError(cutStopTimer(partialContigTimer));
	setStatItem(TM_CONTIG_GENERATION, cutGetTimerValue(partialContigTimer));
	cutilCheckError(cutDeleteTimer(partialContigTimer));

	deallocateMemory(d_ev);
	deallocateMemory(d_levEdge);
	deallocateMemory(d_entEdge);
	deallocateMemory(d_ee);

	resetDevice();
}
/*void transform(){

}*/
void assemble2(	const char * filename, 	//input filename
				const char * output, 	//output filename
				unsigned int readLength,	//readLength
				bool assemble,
				unsigned  int l,		//lmer length
				unsigned int coverage,	//coverage M
				bool errorCorrection,
				unsigned int max_ec_pos,	//ec positions
				unsigned int ec_tuple_size	//ec tuple size
				){


		char * 			readBuffer=NULL;
		EulerVertex * 	ev=NULL;
		EulerEdge 	* 	ee=NULL;
		unsigned int * 	levEdge=NULL;
		unsigned int * 	entEdge=NULL;
		unsigned int  	edgeCount=0;
		unsigned int 	vertexCount=0;
		unsigned int 	readCount=0;


		logMessage(LOG_LVL_MSG, "#l : %d", l);
		setStatItem(NM_LMER_LENGTH, l);

		//input

		readCount=processReads(filename,readLength,&readBuffer);
		if(readBuffer!=NULL){

			if(errorCorrection){
				readCount=doErrorCorrection(&readBuffer,readCount,readLength,ec_tuple_size,max_ec_pos);
			}
			//now we have a new readCount
			//PathSystem P(readCount,readCount*readLength,readCount*readLength);
			if(assemble){
				PathSystem P(1,1,1);
				constructDebruijnGraph(readBuffer,readCount,readLength,l,&ev,&ee,&levEdge, &entEdge, &edgeCount,&vertexCount,P);
				//transform(ev,ee,levEdge,entEdge,edgeCount,vertexCount,P,readBuffer,readCount);
				findEulerTour(ev,ee,levEdge,entEdge,edgeCount,vertexCount,l,output);
				if(ev!=NULL)free(ev);
				if(ee!=NULL)free(ee);
				if(levEdge!=NULL)free(levEdge);
				if(entEdge!=NULL)free(entEdge);
			}
			free(readBuffer);

		}else{
			return;
		}
}
/*
void assemble(const char * fileName, const char * output, unsigned int l,
		unsigned int coverage, unsigned int readLength) {

	EulerVertex * d_ev;
	unsigned int * d_l;
	unsigned int * d_e;
	EulerEdge * d_ee;
	CircuitEdge * d_cg_edge;
	unsigned int ecount;
	unsigned int cg_edgecount;
	unsigned int cg_vertexcount;
	unsigned int * tree;
	unsigned int * d_tree;
	KEY_PTR h_kmerKeys;
	VALUE_PTR h_kmerValues;
	KEY_PTR h_lmerKeys;
	VALUE_PTR h_lmerValues;
	KEY_PTR d_kmerKeys;
	VALUE_PTR d_kmerValues;
	KEY_PTR d_lmerKeys;
	VALUE_PTR d_lmerValues;
	unsigned int bucketCount;
	unsigned int tableLength;
	KEY_PTR d_TK;
	VALUE_PTR d_TV;
	unsigned int * d_bucketSeed;
	unsigned int lmerCount;
	unsigned int kmerCount;
	unsigned int kmerExtractTimer = 0;

	//kmerLength=4;
	logMessage(LOG_LVL_MSG, "#l : %d", l);
	setStatItem(NM_LMER_LENGTH, l);
	cutilCheckError(cutCreateTimer(&kmerExtractTimer));
	cutilCheckError(cutStartTimer(kmerExtractTimer));
//	readLmers(fileName,l,&h_lmerKeys,&h_lmerValues,&lmerCount);
//	readKmers(fileName,l-1,&h_kmerKeys,&h_kmerValues,&kmerCount);

	readLmersKmersCuda(fileName, l, &h_lmerKeys, &h_lmerValues, &lmerCount,
			&h_kmerKeys, &h_kmerValues, &kmerCount, readLength);

//	readLmersKmers(fileName,l,&h_lmerKeys,&h_lmerValues,&lmerCount,&h_kmerKeys,&h_kmerValues,&kmerCount);
	setStatItem(NM_LMER_COUNT, lmerCount);
	setStatItem(NM_KMER_COUNT, kmerCount);
	for (unsigned int i = 0; i < lmerCount; i++) {
		double value;
		if (h_lmerValues[i] < coverage) {
			if (h_lmerValues[i] > 0) {
				h_lmerValues[i] = 1;
			}
		} else {
			value = (double) (h_lmerValues[i]) / (double) (coverage);
			h_lmerValues[i] = (unsigned int) (value + 0.5);
		}
	}
	cutilCheckError(cutStopTimer(kmerExtractTimer));
	setStatItem(TM_KMER_EXTRACTION_TIME, cutGetTimerValue(kmerExtractTimer));
	cutilCheckError(cutDeleteTimer(kmerExtractTimer));

	allocateMemory((void**) &d_lmerKeys, lmerCount * (KEY_SIZE));
	allocateMemory((void**) &d_lmerValues, lmerCount * (VALUE_SIZE));
	allocateMemory((void**) &d_kmerKeys, kmerCount * (KEY_SIZE));
	allocateMemory((void**) &d_kmerValues, kmerCount * (VALUE_SIZE));

	unsigned int hashTableTimer = 0;
	cutilCheckError(cutCreateTimer(&hashTableTimer));
	cutilCheckError(cutStartTimer(hashTableTimer));
	//printKmer(h_idata,mem_size);
	cutilSafeCall(
			cudaMemcpy(d_lmerKeys, h_lmerKeys, lmerCount * (KEY_SIZE),
					cudaMemcpyHostToDevice));
	cutilSafeCall(
			cudaMemcpy(d_lmerValues, h_lmerValues, lmerCount * (VALUE_SIZE),
					cudaMemcpyHostToDevice));
	cutilSafeCall(
			cudaMemcpy(d_kmerKeys, h_kmerKeys, kmerCount * (KEY_SIZE),
					cudaMemcpyHostToDevice));
	cutilSafeCall(
			cudaMemcpy(d_kmerValues, h_kmerValues, kmerCount * (VALUE_SIZE),
					cudaMemcpyHostToDevice));

	createHashTable(d_kmerKeys, d_kmerValues, kmerCount, &d_TK, &d_TV,
			&tableLength, &d_bucketSeed, &bucketCount);
	cutilCheckError(cutStopTimer(hashTableTimer));
	setStatItem(TM_HASHTABLE_CONSTRUCTION, cutGetTimerValue(hashTableTimer));
	cutilCheckError(cutDeleteTimer(hashTableTimer));

	//if(cpu){
//		constructDebruijnGold(d_idata,d_icount,kmerCount,kmerLength,&d_l,&d_e,&d_ee,&d_ev,&vcount,&ecount);
	//}else{
	unsigned int debruijnTimer = 0;

	cutilCheckError(cutCreateTimer(&debruijnTimer));
	cutilCheckError(cutStartTimer(debruijnTimer));

	constructDebruijnGraphDevice(d_lmerKeys, d_lmerValues, lmerCount,
			d_kmerKeys, kmerCount, l, d_TK, d_TV, d_bucketSeed, bucketCount,
			&d_ev, &d_l, &d_e, &d_ee, &ecount);
	cutilCheckError(cutStopTimer(debruijnTimer));
	setStatItem(TM_DEBRUIJN_CONSTRUCTION, cutGetTimerValue(debruijnTimer));
	setStatItem(NM_DEBRUIJN_VTX, kmerCount);
	setStatItem(NM_DEBRUIJN_EDG, ecount);
	cutilCheckError(cutDeleteTimer(debruijnTimer));
//	}

	deallocateMemory(d_lmerKeys);
	deallocateMemory(d_lmerValues);
	deallocateMemory(d_kmerKeys);
	deallocateMemory(d_kmerValues);
	deallocateMemory(d_TK);
	deallocateMemory(d_TV);
	deallocateMemory(d_bucketSeed);
	free(h_lmerKeys);
	free(h_lmerValues);
	free(h_kmerKeys);
	free(h_kmerValues);

	unsigned int eulerTimer = 0;
	cutilCheckError(cutCreateTimer(&eulerTimer));
	cutilCheckError(cutStartTimer(eulerTimer));

	findEulerDevice(d_ev, d_l, d_e, kmerCount, d_ee, ecount, &d_cg_edge,
			&cg_edgecount, &cg_vertexcount, l);
	setStatItem(NM_CIRCUIT_VTX, cg_vertexcount);
	setStatItem(NM_CIRCUIT_EDG, cg_edgecount);
	if (cg_edgecount > 0) {

		unsigned int mstTimer = 0;
		cutilCheckError(cutCreateTimer(&mstTimer));
		cutilCheckError(cutStartTimer(mstTimer));

		CircuitEdge * cg_edge = (CircuitEdge *) (malloc(
				sizeof(CircuitEdge) * cg_edgecount));
		cutilSafeCall(
				cudaMemcpy(cg_edge, d_cg_edge,
						sizeof(CircuitEdge) * cg_edgecount,
						cudaMemcpyDeviceToHost));

		typedef adjacency_list<listS, vecS, undirectedS, no_property,
				property<edge_index_t, int, property<edge_weight_t, int> > > Graph;
		typedef graph_traits<Graph>::edge_descriptor Edge;
		typedef graph_traits<Graph>::vertex_descriptor Vertex;

		int * weights = (int *) malloc(sizeof(int) * cg_edgecount);
		for (unsigned int i = 0; i < cg_edgecount; i++) {
			weights[i] = 1;
		}
		Graph g(cg_vertexcount);
		property_map<Graph, edge_weight_t>::type weightmap = get(edge_weight,
				g);
		property_map<Graph, edge_index_t>::type indexmap = get(edge_index, g);
		for (std::size_t j = 0; j < cg_edgecount; ++j) {
			Edge e;
			bool inserted;
			tie(e, inserted) = add_edge(cg_edge[j].c1, cg_edge[j].c2, g);
			weightmap[e] = weights[j];
			indexmap[e] = (int) j;
		}

		property_map<Graph, edge_index_t>::type index = get(edge_index, g);
		std::vector<Edge> spanning_tree;
		kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));

		tree = (unsigned int *) malloc(
				spanning_tree.size() * sizeof(unsigned int));
		memset(tree, 0, spanning_tree.size() * sizeof(unsigned int));
		unsigned i = 0;
		logMessage(LOG_LVL_DEBUG, "$$Tree Edges");
		for (std::vector<Edge>::iterator ei = spanning_tree.begin();
				ei != spanning_tree.end(); ++ei) {
			tree[i] = index[*ei];
			//logMessage(LOG_LVL_DEBUG,"$edge index %u ",tree[i]);
			i++;
		}


		allocateMemory((void**) &d_tree,
				(unsigned int) spanning_tree.size() * sizeof(unsigned int));
		cutilSafeCall(
				cudaMemcpy(d_tree, tree,
						spanning_tree.size() * sizeof(unsigned int),
						cudaMemcpyHostToDevice));
		cutilCheckError(cutStopTimer(mstTimer));
		setStatItem(TM_SPANNING_TREE, cutGetTimerValue(mstTimer));
		cutilCheckError(cutDeleteTimer(mstTimer));

		unsigned int swipeTimer = 0;
		cutilCheckError(cutCreateTimer(&swipeTimer));
		cutilCheckError(cutStartTimer(swipeTimer));
		executeSwipeDevice(d_ev, d_e, kmerCount, d_ee, ecount, d_cg_edge,
				cg_edgecount, d_tree, (unsigned int) spanning_tree.size());
		cutilCheckError(cutStopTimer(swipeTimer));
		setStatItem(TM_SWIPE_EXECUTION, cutGetTimerValue(swipeTimer));
		cutilCheckError(cutDeleteTimer(swipeTimer));

		deallocateMemory(d_tree);
		deallocateMemory(d_cg_edge);
		free(cg_edge);
		free(tree);
		free(weights);

	}

	cutilCheckError(cutStopTimer(eulerTimer));
	setStatItem(TM_EULER_TOUR, cutGetTimerValue(eulerTimer));
	cutilCheckError(cutDeleteTimer(eulerTimer));


	unsigned int partialContigTimer = 0;
	cutilCheckError(cutCreateTimer(&partialContigTimer));
	cutilCheckError(cutStartTimer(partialContigTimer));
	generatePartialContig(output, d_ev, kmerCount, d_ee, ecount, l);
	cutilCheckError(cutStopTimer(partialContigTimer));
	setStatItem(TM_CONTIG_GENERATION, cutGetTimerValue(partialContigTimer));
	cutilCheckError(cutDeleteTimer(partialContigTimer));

	deallocateMemory(d_ev);
	deallocateMemory(d_l);
	deallocateMemory(d_e);
	deallocateMemory(d_ee);

}*/
void runMemTest() {
	for (int i = 1; i <= 100; i++) {
		unsigned int * d_buffer;
		allocateMemory((void**) &d_buffer, 4 * 1024 * 1024 * ((i % 10) + 1));
		deallocateMemory(d_buffer);
	}
}
void printUsage(char ** argv) {
	printf("Usage : %s <inputFile> <outputfile> <kmerLength>\n", argv[0]);
}


void printArgumentValues(){

	cout<<" input file :"<<vm["input-file"].as<string>()<<"\n";
	cout<<" output file :"<<vm["output-file"].as<string>()<<"\n";
	cout<<" read length :"<<vm["read-length"].as<int>()<<"\n";
	cout<< "Processing options:";
	if(vm.count("error-correction")){
		cout<<" error-correction ";
	}
	if(vm.count("assemble")){
		cout<<"assemble";
	}
	cout<<"\n";

	if(vm.count("error-correction")){
		cout<<"Error correction options:\n";
		cout<<"tuple:"<<vm["tuple"].as<int>()<<" , max-iterations:"<<vm["max-iterations"].as<int>()<<"\n";
	}
	if(vm.count("assemble")){
			cout<<"Assembly parameters:\n";
			cout<<"coverage:"<<vm["coverage"].as<int>()<<" , lmer:"<<vm["lmer"].as<int>()<<" , block-size:"<<vm["block-size"].as<int>()<<"\n";
	}
	if(vm.count("verbose")){
		cout<<"using log level:"<<LOG_LEVEL<<"\n";
	}

}
//namespace po=boost::program_options;
void parseArguments(int argc, char ** argv) {



	try{
		po::options_description general("General arguments");
		general.add_options()("help,h", "display this help");


		po::options_description required("Required arguments");
		required.add_options()
				("input-file,i", po::value<string>(),":Input file in FASTA format")
				("output-file,o", po::value<string>(),":Output file  (generated in FASTA format)")
				("read-length,r",po::value<int>(), ":Read Length for input set of reads");

		po::options_description ecArguments("Error Correction Parameters");
		ecArguments.add_options()
			("error-correction,e",":Enable error correction")
			("tuple,t",po::value<int>(), ":tuple size for error correction")
			("max-iterations,m",po::value<int>()->default_value(1), ":Error correction iterations (1-5)");

		po::options_description asmArguments("Assembly Parameters");
		asmArguments.add_options()
				("assemble,asm",":Enable Assembly")
				("lmer,l",po::value<int>()->default_value(16), ":tuple size for debruijn graph")
				("block-size,b", po::value<int>()->default_value(512),":block size for CUDA execution ")
				("coverage,c", po::value<int>()->default_value(20),":Read Coverage");
		required.add(ecArguments).add(asmArguments);

		po::options_description optional("Optional arguments");
		optional.add_options()
				("verbose,v", po::value<int>(),":verbose level (0=off, 9=full)")
				("log-file,g", po::value<string>(),":log file for messages (default stdout)")
				("stat-format,f",po::value<int>(), ":statistic format (default xml)");
				//("hash-impl,m",po::value<char>(),  ":hash table implementation.\n"
					//						"     g : \tgoogle-sparse-hash\n"
						//					"     b : \tboost-unordered-set\n");


		po::options_description all("EulerCUDA");
		all.add(general).add(required).add(optional);

		po::store(po::parse_command_line(argc, argv, all), vm);
		po::notify(vm);

		if ( vm.count("help")) {
			cout << all << "\n";
			exit(0);
		}
		//validations
		if (!vm.count("input-file") || !vm.count("output-file")	|| !vm.count("read-length")	||( !vm.count("assemble") && !vm.count("error-correction"))) {
			cout << "Required argument missing\n" << all << "\n";
			exit(0);
		}
		if( vm.count("assemble")){
			if( !vm.count("lmer") ||  !vm.count("block-size")  || !vm.count("coverage")){
				cout<< "Assembly parameters are missing\n"<<all<<"\n";
				exit(0);
			}
		}
		if( vm.count("error-correction")){
			if( !vm.count("tuple") ||  !vm.count("max-iterations") ){
				cout<< "Error Correction parameters are missing\n"<<all<<"\n";
				exit(0);
			}
		}

		if(vm.count("verbose")){
			LOG_LEVEL=vm["verbose"].as<int>();
		}

		//printArgumentValues();
		//exit(0);
	}
	catch(std::exception& e){
		std::cerr << "Error: " << e.what() << "\n";
		exit(0);
	}
	catch(...)    {
		std::cerr << "Unknown error!" << "\n";
		exit(0);
	}
}

int main(int argc, char * argv[]) {


	int computeGold = 0;

	const char * inputFileName;
	const char * outputFileName;
	int readLength;


	bool assemble=false;
	int lmerLength = 10;
	int blockSize = DEFAULT_BLOCK_SIZE;
	int coverage=20;

	bool errorCorrection=false;
	int max_ec_pos=1;
	int ec_tuple_size=26;

	parseArguments(argc,argv);

	inputFileName=vm["input-file"].as<string>().c_str();
	outputFileName=vm["output-file"].as<string>().c_str();
	readLength=vm["read-length"].as<int>();

	if(vm.count("error-correction")){
		errorCorrection=true;
		max_ec_pos=vm["max-iterations"].as<int>();
		ec_tuple_size=vm["tuple"].as<int>();
	}
	if(vm.count("assemble")){
		assemble=true;
		coverage=vm["coverage"].as<int>();
		blockSize=vm["block-size"].as<int>();
		lmerLength=vm["lmer"].as<int>();
		setBlockSize(blockSize);
	}

	DEBUG_CALL(LOG_LEVEL=LOG_LVL_DETAIL);
	/****************************/
	initMemList();
	initDevice();


	if (computeGold == 0 || computeGold == 2) {
		unsigned int timerGPU = 0;
		cutilCheckError(cutCreateTimer(&timerGPU));
		cutilCheckError(cutStartTimer(timerGPU));
		assemble2(inputFileName, outputFileName, readLength, assemble, lmerLength, coverage,errorCorrection, max_ec_pos,ec_tuple_size);
		cutilCheckError(cutStopTimer(timerGPU));
		logMessage(LOG_LVL_MSG, "#Total Time : %f",cutGetTimerValue(timerGPU));
		cutilCheckError(cutDeleteTimer(timerGPU));
	}
	if (computeGold == 1 || computeGold == 2) {
		assembleGold(inputFileName, outputFileName, lmerLength, coverage);
	}
	writeStat(stdout, 0);

	cleanupMemList();
	cudaThreadExit();
	//cutilExit(argc, argv);
	return 0;
}
