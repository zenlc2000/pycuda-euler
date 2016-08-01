#ifndef SEQUENCE_H
#define SEQUENCE_H


typedef  struct Read{
	unsigned int length;
	unsigned char * data;
	struct Read * next;
} Read;

void allocateRead(Read ** read);
#endif //SEQUENCE_H