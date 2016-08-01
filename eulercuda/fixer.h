#ifndef FIXER_H
#define FIXER_H

extern "C"
unsigned int errorCorrection(char * in_reads,unsigned int readCount,
		unsigned int readLength,		 char ** correctedReads,unsigned int tuple_size) ;
#endif
