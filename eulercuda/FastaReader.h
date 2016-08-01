
#ifndef FASTA_READER_H
#define FASTA_READER_H

#include "Sequence.h"



typedef struct KmerReader {
	FILE * fptr;
	int status;
	unsigned char * kmer;
	unsigned int k;
}KmerReader;
typedef bool   ( *fastaReaderCallback)(Read ** start,unsigned int count,void *  param );
 long readFASTA(const char * path, Read** head, fastaReaderCallback callBack, int threshold,void * param);
 void deleteList(Read ** head);

 KmerReader * createKmerReader(const char * path,int kmerLength);
 int getNext(KmerReader* kmerReader);
 void destroyKmerReader(KmerReader * kmerReader);

//returns number of actual reads of fixed length with buffer set to reads
unsigned int getReads(FILE * fptr,char * buffer, const unsigned int buffSize, const unsigned int  maxReadLength, const unsigned int numReads);

 /*
 * FASTAReader* createReader(File);
 * void destroyReader(FastraReader* );
 * getNext() 
 * 
 getNext(KmerReader){
 }
 *
 */

#endif //FAST_READER_H
