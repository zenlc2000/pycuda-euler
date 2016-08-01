#include <stdlib.h>
#include <malloc.h>
#include "Sequence.h"


void allocateRead(Read ** read){
	(*read)=(Read *)malloc(sizeof(Read));
	//should handle malloc error here

	//init
	(*read)->length=0;
	(*read)->data=NULL;
	(*read)->next=NULL;
}