#ifndef FIXER_CUH
#define FIXER_CUH

extern "C"
void getMutationScores(	char * in_read,
						char * mutationMask,
						unsigned int readLength,
						unsigned int lmerLength,
						unsigned int M,
						KEY_PTR TK,
						VALUE_PTR TV,
						unsigned int * bucketSize,
						unsigned int bucketCount,
						/*out param*/
						unsigned int * mutation ,//out,
						unsigned int * buffer
			);
			
			
extern "C"
void getAccumulatedScores(	unsigned int * mutation,
							unsigned int readLength,
							unsigned int l,
								/*out*/
							unsigned int * mutationScore
							);


extern "C"
void getMutationScores2(	char * in_read,
						char * mutationMask,
						unsigned int readLength,
						unsigned int lmerLength,
						unsigned int M,
						KEY_PTR TK,
						VALUE_PTR TV,
						unsigned int * bucketSize,
						unsigned int bucketCount,
                                                unsigned int readCount,
						/*out param*/
						unsigned int * mutation ,//out,
						unsigned int * buffer
			);
			
extern "C"
void getMutationScores3(	char * in_read,
						char * mutationMask,
						unsigned int readLength,
						unsigned int lmerLength,
						unsigned int M,
						KEY_PTR TK,
						VALUE_PTR TV,
						unsigned int * bucketSize,
						unsigned int bucketCount,
                                                unsigned int readCount,
						/*out param*/
						unsigned int * mutation ,//out,
						unsigned int * buffer
			);			
extern "C"
void getAccumulatedScores2(	unsigned int * mutation,
							unsigned int readLength,
							unsigned int l,
                                                        unsigned int readCount,       
								/*out*/
							unsigned int * mutationScore
							);
							

// n x 4 -> n
extern "C" 
void getBestMutation2(unsigned int * d_mutationScore,
        unsigned int readLength, unsigned int l,
        unsigned int readCount,
        /*out*/
        unsigned int * d_mutationStep) ;

//n -> 1
extern "C" 
void getBestFinalMutation2(unsigned int * d_mutationStep,
        unsigned int readLength, unsigned int l,
        unsigned int readCount,
        /*out*/
        unsigned int * d_bestMutationPos,
        unsigned int * d_bestMutationIdx) ;


#endif