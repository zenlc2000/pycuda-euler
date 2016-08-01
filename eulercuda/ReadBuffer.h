#ifndef READ_BUFFER_H
#define READ_BUFFER_H

#include "Sequence.h"

#define MAX_BUFFER 160

typedef struct InputBuffer{
	int count;
	int prevCount;
	InputBuffer * next;
	unsigned char buffer[MAX_BUFFER];
}InputBuffer;


int appendBuffer(InputBuffer ** tail,char c,int prevCount);
bool commitBuffer(InputBuffer ** head,InputBuffer ** tail,Read** read,int prevCount);
void allocateBuffer(InputBuffer ** buffer);
#endif