#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <memory.h>
#include <string.h>
#include "Sequence.h"
#include "ReadBuffer.h"
#include "FastaReader.h"


#define FINAL_STATE 0
#define INIT_STATE 0
#define STATE_A 1
#define STATE_B 2
#define STATE_C 3
#define SYMBOL_GT  0 //>
#define SYMBOL_SC  1 //;
#define SYMBOL_CHR 2 //any char
#define SYMBOL_NL  3 //newline

#define FASTA_READER_BUFFER_SIZE	1024*1024

int  transitions[4][4]={
	{STATE_A,STATE_A,STATE_B,STATE_C},
	{STATE_B,STATE_A,STATE_B,STATE_C},
	{STATE_C,STATE_A,STATE_B,STATE_C},
	{INIT_STATE,INIT_STATE,INIT_STATE,INIT_STATE}
			};

bool isChar(char ch){
	if((ch>=65 && ch<=90) || (ch>=97 && ch<=122))
			return true;
	else return false;
}
int getSymbol(char ch){
	switch(ch){
		case '>':
			return SYMBOL_GT;
		case ';':
			return SYMBOL_SC;
		case '\n':
		case '\r':
			return SYMBOL_NL;
		default :
			return SYMBOL_CHR;
	}
}

/*

long readFASTA(const char * path,  Read** head){
	
	return 0;
}*/

/*
threshold=0 return all;s
*/
long readFASTA(const char * path,  Read** head, fastaReaderCallback callback, int threshold, void * param){

	FILE * file;

	if( head ==NULL) return -1;

	file=fopen(path,"r");
	if(file==NULL){
		printf("Error Opening File");
		return -1;
	}	
	else{
		allocateRead(head); //create dummy
		Read * currentRead=*head;
		InputBuffer * bufferHead=NULL;
		InputBuffer * bufferTail=NULL;
		int readCount=0;
		int subCount=0;
		char c=fgetc(file);
		int state=INIT_STATE;
		//int nextState=INIT_STATE;
		int prevCount=0;
		int symbol=0;
		while(c!=EOF){
			symbol=getSymbol(c);
				switch(state){
					case INIT_STATE:
						switch(symbol){
							case SYMBOL_GT:
								//commit previous buffers
								if(commitBuffer(&bufferHead,&bufferTail,&currentRead,prevCount)){
									//number reads successfully read so far
									readCount++;
									subCount++;
								}
								if(subCount>=threshold && threshold!=0){
									//call callback
									if(callback!=NULL){
										if(callback(&((*head)->next),subCount,param)){
											Read ** midRead=&((*head)->next);
											while(subCount!=0){
												Read * tmp=*midRead;
												*midRead=(*midRead)->next;
												free(tmp->data);
												free(tmp);
												subCount--;
											}
											currentRead=*head;
										}
									}
								}
								 //malloc read structure
								allocateRead(&(currentRead->next));							
								 //add to list
								currentRead=currentRead->next;							
								//allocate buffer
								allocateBuffer(&bufferHead);							
								//set tail pointer
								bufferTail=bufferHead;
								//set previous buffer count
								prevCount=0;								
								break;

							case SYMBOL_CHR:
								//copy to the read buffer
								prevCount=appendBuffer(&bufferTail,c,prevCount);
								break;
							case SYMBOL_NL:
							case SYMBOL_SC:
								break;		//do nothing
						}														 
						break;
					case STATE_A:
						break;		//do nothing 					
					case STATE_B:
						break;		//do nothing
					case STATE_C:
						switch(symbol){
							case SYMBOL_GT:
							case SYMBOL_SC:
								//report error here
								break;						
							case SYMBOL_CHR:
								//copy to the read buffer
								prevCount=appendBuffer(&bufferTail,c,prevCount);
								break;
						case SYMBOL_NL:
								break;
						}
						break;
					}
			state=transitions[symbol][state];
			c=fgetc(file);
		}
		if(commitBuffer(&bufferHead,&bufferTail,&currentRead,prevCount)){
			//number reads successfully read so far
			readCount++;
			subCount++;
		}
		if(subCount>0  && threshold!=0){
			//call callback
			if(callback!=NULL){
				if(callback(&((*head)->next),subCount,param)){
					Read ** midRead=&((*head)->next);
					while(subCount!=0){
						Read * tmp=*midRead;
						*midRead=(*midRead)->next;
						free(tmp->data);
						free(tmp);
						subCount--;
					}
					currentRead=*head;
				}
			}
		}
		Read * tmp=(*head);
		(*head)=(*head)->next;
		free(tmp);
		fclose(file);
		return readCount;
	}
}

 void deleteList(Read ** head){
	 if(head!=NULL && *head!=NULL){
		 Read * tmp;
		 while(*head!=NULL){
			 tmp=*head;
			 *head=(*head)->next;
			 free(tmp->data);
			 free(tmp);
		 }
	 }
}
/***2nd implementation **/
 void setupState(KmerReader * kmerReader) {
	 char c=fgetc(kmerReader->fptr);
	 size_t bytesRead=0;
	 if(c=='>'){		
		 while(c!='\n' && c!='\r')  c=fgetc(kmerReader->fptr);
		 while(c=='\n' || c=='\r') c=fgetc(kmerReader->fptr);
		 kmerReader->kmer[0]=c;
		 bytesRead=fread(kmerReader->kmer+1,sizeof(unsigned char),kmerReader->k-2,kmerReader->fptr);
	 }
	 else{
		 kmerReader->status=-1;
	 }
 }
 KmerReader * createKmerReader(const char * path,int kmerLength){
	 KmerReader * kmerReader=NULL;
	 /*init*/
	 kmerReader=(KmerReader * ) malloc(sizeof(KmerReader));
	 kmerReader->k=kmerLength;
	 kmerReader->kmer=(unsigned char*) malloc(sizeof(char)*(kmerLength+1));
	 kmerReader->kmer[kmerLength]=0;
	 kmerReader->fptr=fopen(path,"r");
	 kmerReader->status=0;
	// setupState(kmerReader);
	 /*read*/
	 return kmerReader;
 }
 int getNext(KmerReader* kmerReader){
	char c;
	size_t bytesRead=0;
	 c=fgetc(kmerReader->fptr);
	 if( c==EOF) {
		 kmerReader->status=-1;
		 return -1;
	 }
	 while(c=='\n' || c=='\r')c=fgetc(kmerReader->fptr);
	 if( c==EOF) {
		 kmerReader->status=-1;
		 return -1;
	 }
	 if( c=='>'){
		 while(c !='\n' && c != '\r') c=fgetc(kmerReader->fptr);
		 while(c=='\n' || c=='\r')c=fgetc(kmerReader->fptr);
		 kmerReader->kmer[0]=c;
		 bytesRead=fread(kmerReader->kmer+1,sizeof(unsigned char),kmerReader->k-1,kmerReader->fptr);

	 }else {
		 memmove(kmerReader->kmer,kmerReader->kmer+1,kmerReader->k-1);
		 kmerReader->kmer[kmerReader->k-1]=c;
	 }
	 return -1;
 }
 void destroyKmerReader(KmerReader * kmerReader){
	 fclose(kmerReader->fptr);
	 free(kmerReader->kmer);
	 free(kmerReader);
 }

/****3rd implementation of fasta reader for fixed length reads,*/
unsigned int getReads(FILE* fptr,char * buffer, const unsigned int buffSize, const unsigned int  maxReadLength, const unsigned int numReads){

	/*
	* Todo : support for variable length reads, param check , validity, etc.
	*/
	memset(buffer,0,buffSize);
	/*
	char * tBuff;	
	char c;
	unsigned int cc=0;
	unsigned int readCount=0;
	tBuff=buffer;
	c=fgetc(fptr);
	while(c!=EOF && readCount<numReads){
		while((c=='\n' || c=='\r') && c!='>')c=fgetc(fptr);
		if(c=='>'){ //next read
			cc=0;
			while(c!='\n' && c!='\r') c=fgetc(fptr);   //eat identifier info
			while(c=='\n'  || c=='\r') c=fgetc(fptr); //skip to read data
			while(c!='>' && c!=EOF){
				while(c!='\n' && c!='\r') {
					*tBuff=c;
					tBuff++;
					cc++;
					c=fgetc(fptr);		
				}
				while(c=='\n'|| c=='\r') c=fgetc(fptr);
			}
			readCount++;
		}
			
	}	
	if(c=='>') fseek(fptr,-1,SEEK_CUR);
	return readCount;
	*/

	/*using fgets*/
	char * tBuff;
	char * inputBuff;
	char * c;
	unsigned int cc=0;
	unsigned int readCount=0;

	tBuff=buffer;
	inputBuff=(char *)malloc(FASTA_READER_BUFFER_SIZE);
	memset(inputBuff,0,FASTA_READER_BUFFER_SIZE);

	c=fgets(inputBuff,FASTA_READER_BUFFER_SIZE,fptr);
	do{

		if(c!=NULL && *c =='>'){//new read
			c=fgets(inputBuff,FASTA_READER_BUFFER_SIZE,fptr);
			do{
				while(*c!='\n' && *c!='\r' && *c!='\0') { //copy
					*tBuff=*c;
					tBuff++;
					c++;
				}
				c=fgets(inputBuff,FASTA_READER_BUFFER_SIZE,fptr);//read next line
			}while(c!=NULL && *c!='>' && *c!= '\n' && *c!='\r'); //loop till next read
			readCount++;	//parsed one read.
		}else if(c!=NULL && *c=='\n'){
			c=fgets(inputBuff,FASTA_READER_BUFFER_SIZE,fptr);
		}
	}while(c!=NULL && readCount<numReads);
	if(c!=NULL && *c=='>') fseek(fptr,-strlen(inputBuff),SEEK_CUR);
	free(inputBuff);
	return readCount;

}
