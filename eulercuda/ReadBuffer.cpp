#include <stdlib.h>
#include <malloc.h>
#include <memory.h>
#include "ReadBuffer.h"



int appendBuffer(InputBuffer ** tail,char c,int prevCount){
	int length=-1;
	if(*tail==NULL ) return -1;
	length=prevCount;
	if((*tail)->count==MAX_BUFFER){ //if not sufficient space
			(*tail)->next=(InputBuffer *)malloc(sizeof(InputBuffer));
			length=length+(*tail)->count;
			(*tail)=(*tail)->next;
			(*tail)->count=0;
		}
	(*tail)->buffer[(*tail)->count]=c;
	(*tail)->count++;
	return length;
}

bool commitBuffer(InputBuffer ** head,InputBuffer ** tail,Read** read,int prevCount){
	if(*tail !=NULL){
		InputBuffer * temp;
		(*read)->length=(*tail)->count+prevCount;
		(*read)->data=(unsigned char *)malloc(sizeof(char)*((*read)->length));
		while((*head)!=NULL){
			int index=0;
			memcpy(((*read)->data)+index,(*head)->buffer,(*head)->count);
			index+=(*head)->count;
			temp=(*head);
			(*head)=(*head)->next;
			free(temp);
		}
		return true;
	}
	return false;
}

void allocateBuffer(InputBuffer ** buffer){
	(*buffer)=(InputBuffer *) malloc(sizeof(InputBuffer));
	//malloc error here

	//init
	(*buffer)->count=0;
	(*buffer)->next=NULL;
}