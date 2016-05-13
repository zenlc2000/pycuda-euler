#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>

#define FASTA_READER_BUFFER_SIZE	1024 * 1024


unsigned int getReads(FILE* fptr,char * buffer, const unsigned int buffSize, const unsigned int  maxReadLength, const unsigned int numReads)
{

	/*
	* Todo : support for variable length reads, param check , validity, etc.
	*/
	memset(buffer,0,buffSize);

	/*using fgets*/
	char * tBuff;
	char * inputBuff;
	char * c;
	unsigned int cc = 0;
	unsigned int readCount = 0;

	tBuff = buffer;
	inputBuff = (char *)malloc(FASTA_READER_BUFFER_SIZE);
	memset(inputBuff,0,FASTA_READER_BUFFER_SIZE);

	c = fgets(inputBuff,FASTA_READER_BUFFER_SIZE,fptr);
	do
 	{

		if(c!=NULL && *c =='>')
   		{//new read
			c = fgets(inputBuff,FASTA_READER_BUFFER_SIZE,fptr);
			do
			{
				while(*c!='\n' && *c!='\r' && *c!='\0') 
        		{ //copy
					*tBuff = *c;
					tBuff++;
					c++;
				}
				c = fgets(inputBuff,FASTA_READER_BUFFER_SIZE,fptr);//read next line
			}
      		while(c != NULL && *c != '>' && *c != '\n' && *c != '\r'); //loop till next read
			readCount++;	//parsed one read.
		}
	    else if(c != NULL && *c == '\n')
	    {
				c = fgets(inputBuff,FASTA_READER_BUFFER_SIZE,fptr);
		}
	} while(c!=NULL && readCount<numReads);
	if(c != NULL && *c == '>') fseek(fptr,-strlen(inputBuff),SEEK_CUR);
	free(inputBuff);
	return readCount;

}


unsigned int processReads(	const char * filename,		//input file
					unsigned int readLength,	//read length
					char **	readBuffer			//out put buffer
					)
{

	unsigned int readCount = 0;
	char * tempBuff = NULL;
	FILE * fptr = NULL;
	//allocate very big buffer and truncate it later on, we might use linked lst


	unsigned int expectedReadCount = 4 * 1024 * 1024;// 4 million reads
	unsigned int maxBufferSize = sizeof(char)*readLength*expectedReadCount;

	unsigned int processeReadsTimer = 0;

	tempBuff=(char *)malloc(maxBufferSize);

	fptr = fopen(filename,"r");
	if(fptr != NULL)
	{
		readCount = getReads(fptr,tempBuff,maxBufferSize,readLength,expectedReadCount);
		if(expectedReadCount>readCount)
		{
			unsigned int newBufferSize = readCount * readLength;
			*readBuffer = (char *)malloc(newBufferSize);
			memcpy(*readBuffer,tempBuff,newBufferSize);
		}
		else
		{
			//need larger buffer,
			*readBuffer = NULL;
		}
	}
	else
	{
		//log error opening file
	}

	if(tempBuff!=NULL) free(tempBuff);
	if(fptr != NULL) fclose(fptr);
	return readCount;
}

int main(int argc, char *argv[]) 
{
	char *readBuffer = NULL;
	const char * filename = "../data/Ecoli_raw.fasta";
	unsigned int readCount = 0;
	
	readCount = processReads(filename, 17, &readBuffer);
	
	printf("readCount = %d\n",readCount);
	printf("%s",readBuffer[0]);
}