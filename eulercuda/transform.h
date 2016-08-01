
#ifndef TRANSFORM_H_
#define TRANSFORM_H_

extern "C"
void transform(EulerVertex * ev, EulerEdge * ee, unsigned int * levEdge, unsigned int * entEdge, unsigned int & edgeCount,
		unsigned int & vertexCount, PathSystem & P, char * reads, unsigned int readCount);

#endif /* TRANSFORM_H_ */
