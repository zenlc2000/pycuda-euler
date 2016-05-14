import sys
import argparse
import pyencode as enc
import numpy as np
from numpy import array
import logging
import logging.config
from parse_fasta import Fasta

def parse_fastq ( filename ):
    """
    Read fastq formatted <filename> and return a dictionary of
    read_name : read
    """
    file = open ( filename )
    result = {}
    current_name = None
    for i, line in enumerate ( file ):
        if i % 4 == 0:
            current_name = line.rstrip ( '\n' )
        if i % 4 == 1:
            result[ current_name ] = line.rstrip ( '\n' )
        if i % 4 == 3:
            print ( 'quality is ' + line.rstrip ( '\n' ) )
    return result


def read_fastq ( filename ):
    """
    Read fastq formatted <filename> and return a list of reads
    """
    with open ( filename, "r" ) as infile:
        result = [ ]
        for i, line in enumerate ( infile ):
            if i % 4 == 1:
                result.append ( line.rstrip ( '\n' ) )
        return result


def doErrorCorrection ( readBuffer, readCount, ec_tuple_size, max_ec_pos ):
    return readCount


def readLmersKmersCuda ( readBuffer, readLength, readCount, lmerLength, lmerKeys, lmerValues, lmerCount, kmerKeys,
                         kmerValues, kmerCount ):
    """
    char * d_reads = NULL;
    KEY_PTR h_lmersF = NULL;
    KEY_PTR h_lmersR = NULL;
    KEY_PTR d_lmers = NULL;
    KEY_PTR h_pkmersF = NULL;
    KEY_PTR h_pkmersR = NULL;
    KEY_PTR h_skmersF = NULL;
    KEY_PTR h_skmersR = NULL;
    KEY_PTR d_pkmers = NULL;
    KEY_PTR d_skmers = NULL;
    unsigned int readProcessed=0;
    unsigned int kmerGPUEncTimer = 0;
    unsigned int kmerExtractTimer=0;

    typedef dense_hash_map<KEY_T, VALUE_T> map;
    map kmerMap(readLength*readCount);
    map lmerMap(readLength*readCount);
    """

    # bufferSize = buffer.size
    kmerMap = {}
    lmerMap = {}
    # buffer = np.fromstring('\n'.join(readBuffer), count=len(readBuffer), dtype=np.str)
    buffer = np.array(readBuffer, dtype = str)
    nbr_values = buffer.size * buffer.dtype.itemsize
    d_lmers = np.empty(buffer.size, dtype = np.uint64)
    d_pkmers = np.empty_like ( d_lmers )
    d_skmers = np.empty_like ( d_lmers )
    h_lmersF = np.empty_like ( d_lmers )
    h_pkmersF = np.empty_like ( d_lmers )
    h_skmersF = np.empty_like ( d_lmers )
    h_lmersR = np.empty_like ( d_lmers )
    h_pkmersR = np.empty_like ( d_lmers )
    h_skmersR = np.empty_like ( d_lmers )

    CUDA_NUM_READS = 1024 * 32
    if readCount < CUDA_NUM_READS:
        readToProcess = readCount
    else:
        readToProcess = CUDA_NUM_READS
    kmerBitMask = 0

    bufferSize = sum(sys.getsizeof(seq) for seq in readBuffer)
    entriesCount = readLength * readCount
    readLength = [len(seq) for seq in readBuffer]

    for _ in range ( 0, (lmerLength - 1) * 2 ):
        kmerBitMask = (kmerBitMask << 1) | 1
        # print("kmerBitMask = " + str(kmerBitMask))
    readProcessed = 0
    # Originally a loop slicing readBuffer into chunks then process each chunk
    logging.info("Start encoding")
    while readProcessed < readCount:
        enc.encode_lmer_device(buffer, bufferSize, readCount, d_lmers, readLength, lmerLength, readCount)

        enc.compute_kmer_device(d_lmers,d_pkmers, d_skmers, kmerBitMask)
        enc.compute_lmer_complement_device(buffer, bufferSize, readCount, d_lmers, readLength, lmerLength, readCount)

        logging.info("Finished with Encoder.")

        lmerEmpty, kmerEmpty = 0, 0

        # Here he fills the kmerMap and lmerMap with a nested for loop
        # for j in range(readToProcess):
        #     for i in range(validLmerCount):
        for index in range ( readToProcess ):
            # index = j * readLength + i
            kmerMap[ h_pkmersF[ index ] ] = 1
            kmerMap[ h_skmersF[ index ] ] = 1
            kmerMap[ h_pkmersR[ index ] ] = 1
            kmerMap[ h_skmersR[ index ] ] = 1

            if h_lmersF[ index ] == 0:
                lmerEmpty += 1
            else:
                if lmerMap.get ( h_lmersF[ index ] ) == None:
                    lmerMap[ h_lmersF[ index ] ] = 1
                else:
                    lmerMap[ h_lmersF[ index ] ] += 1
            if h_lmersR[ index ] == 0:
                lmerEmpty += 1
            else:
                if lmerMap.get ( h_lmersR[ index ] ) == None:
                    lmerMap[ h_lmersR[ index ] ] = 1
                else:
                    lmerMap[ h_lmersR[ index ] ] += 1
        readProcessed += readToProcess
        readToProcess = readCount - readToProcess
        if readCount < CUDA_NUM_READS:
            readToProcess = readCount
        else:
            readToProcess = CUDA_NUM_READS
            # End of chunking loop

    kmerCount = len ( kmerMap ) + kmerEmpty
    # TODO: Log message with kmer count
    print ( 'kmer count = ' + kmerCount )

    # kmerKeys = []
    # kmerValues = []
    #
    for index, k, v in enumerate ( kmerMap.items ( ) ):
        kmerKeys[ index ] = k
        kmerValues[ index ] = index

    # original code has if below, but I don't see that it will ever be executed
    #     if (kmerEmpty > 0){
    #     ( * kmerKeys)[index]=0;
    #     ( * kmerValues)[index]=index;
    #     }

    lmerCount = len ( lmerMap ) + lmerEmpty
    # lmerKeys = []
    # lmerValues = []
    #
    lmerKeys = lmerMap.keys ( )
    lmerValues = lmerMap.values ( )

    if lmerEmpty > 0:
        lmerKeys[ len ( lmerMap ) ] = 0
        lmerValues[ len ( lmerMap ) ] = lmerEmpty

    return lmerCount, kmerCount


def constructDebruijnGraph ( readBuffer, readCount, readLength, lmerLength, evList, eeList, levEdgeList, entEdgeList,
                             edgeCountList, vertexCountList ):
    """
    ///variables

    KEY_PTR			h_lmerKeys =NULL;
    VALUE_PTR 		h_lmerValues= NULL;
    KEY_PTR			d_lmerKeys =NULL;
    VALUE_PTR 		d_lmerValues= NULL;
    unsigned int 	lmerCount=0;
    KEY_PTR 		h_kmerKeys=NULL;
    VALUE_PTR 		h_kmerValues=NULL;
    KEY_PTR 		d_kmerKeys=NULL;
    VALUE_PTR 		d_kmerValues=NULL;
    unsigned int 	kmerCount=0;
    KEY_PTR			d_TK=NULL;
    VALUE_PTR		d_TV=NULL;
    unsigned int 	tableLength=0;
    unsigned int	bucketCount=0;
    unsigned int *	d_bucketSize=NULL;

    unsigned int coverage =20;

    EulerVertex * d_ev=NULL;
    EulerEdge 	* d_ee=NULL;
    unsigned int * d_levEdge=NULL;
    unsigned int * d_entEdge=NULL;

    """
    h_lmerKeys = [ ]
    h_lmerValues = [ ]

    lmerCount = 0
    h_kmerKeys = [ ]
    h_kmerValues = [ ]
    d_kmerKeys = [ ]
    d_kmerValues = [ ]
    kmerCount = 0
    d_TK = [ ]
    d_TV = [ ]
    tableLength = 0
    bucketCount = 0
    d_bucketSize = [ ]

    coverage = 20
    d_ev = [ ]
    d_ee = [ ]
    d_levEdge = [ ]
    d_entEdge = [ ]



    # May need to return unpacked tuple of integer variables
    lmerCount, kmerCount = readLmersKmersCuda ( readBuffer, readLength, readCount, lmerLength, h_lmerKeys, h_lmerValues,
                                                lmerCount, h_kmerKeys,
                                                h_kmerValues, kmerCount )
    # initDevice()
    # setStatItem(NM_LMER_COUNT, lmerCount);
    # setStatItem(NM_KMER_COUNT, kmerCount);



    # TODO: return d_TK,  d_TV, tableLength,  d_bucketSize,  bucketCount)
    # TODO: Test pygpuhash
    # pygpuhash.create_hash_table(d_kmerKeys, d_kmerValues, kmerCount)


#    constructDebruijnGraphDevice(d_lmerKeys, d_lmerValues, lmerCount,
#        d_kmerKeys, kmerCount, l, d_TK, d_TV, d_bucketSize, bucketCount,
#        & d_ev, & d_levEdge, & d_entEdge, & d_ee, edgeCount);


# d_bucketSeed needs to

# pydebruijn.construct_Debruijn_Graph_Device(d_lmerKeys, d_lmerValues,lmerCount,
#       d_kmerKeys,kmerCount,l,d_TK, d_TV,d_bucketSize,bucketCount)

# TODO: Copy graph back from device
# *vertexCount = kmerCount;
# *ev=(EulerVertex *)malloc(sizeof(EulerVertex)* (*vertexCount));
# *ee=(EulerEdge *)malloc(sizeof(EulerEdge)* (*edgeCount));
# *levEdge=(unsigned int *)malloc(sizeof(unsigned int)* (*edgeCount));
# *entEdge=(unsigned int * )malloc(sizeof(unsigned int)*(*edgeCount));

# cutilSafeCall(cudaMemcpy(*ev, d_ev, sizeof(EulerVertex) * (*vertexCount),cudaMemcpyDeviceToHost));
# cutilSafeCall(cudaMemcpy(*ee, d_ee, sizeof(EulerEdge)*(*edgeCount),cudaMemcpyDeviceToHost));
# cutilSafeCall(cudaMemcpy(*levEdge, d_levEdge, sizeof(unsigned int)*(*edgeCount),cudaMemcpyDeviceToHost));
# cutilSafeCall(cudaMemcpy(*entEdge, d_entEdge, sizeof(unsigned int)*(*edgeCount),cudaMemcpyDeviceToHost));






# /*variables*/
# EulerVertex * d_ev=NULL;
# EulerEdge 	* d_ee=NULL;
# unsigned int * d_levEdge=NULL;
# unsigned int * d_entEdge=NULL;
#
# CircuitEdge * d_cg_edge=NULL;
# unsigned int cg_edgecount=0;
# unsigned int cg_vertexcount=0;
# unsigned int * tree=NULL;
# unsigned int * d_tree;
# unsigned int treeSize=0;
# /* Timers*/
# unsigned int eulerTimer = 0;
# unsigned int mstTimer = 0;
# unsigned int swipeTimer = 0;
# unsigned int partialContigTimer = 0;


def findEulerTour ( evList, eeList, levEdgeList, entEdgeList, edgeCount, vertexCount, lmerLength, outfile ):
    # TODO: Figure out what to do with these variables
    # findEulerDevice(d_ev, d_levEdge, d_entEdge, vertexCount, d_ee, edgeCount, & d_cg_edge,& cg_edgecount, & cg_vertexcount, l);
    d_ev = array ( evList )
    d_levEdge = array ( levEdgeList )
    d_entEdge = array ( entEdgeList )
    d_ee = array ( eeList )

    # d_cg_edge, cg_vertexcount, cg_edgecount MAY be the output variables.
    pyeulertour.find_euler_device ( d_ev, d_levEdge, d_entEdge, vertexCount, d_ee )
    # need to get cg_edgecount, d_cg_edge back
    if cg_edgecount > 0:
        cg_edge = array ( d_cg_edge )
        treeSize = pyeulertour.find_spanning_tree ( cg_edge, cg_edgecount, cg_vertexcount )


def assemble2 ( infile, outfile, lmerLength, errorCorrection, max_ec_pos, ec_tuple_size ):
    """
    Do the assemble
    """
    # TODO: figure out logging
    # TODO: Unit testing
    logging.getLogger('eulercuda.assemble2')
    # for performance reasons, may want to make these Numpy arrays

    # char * 		readBuffer=NULL;
    # EulerVertex * 	ev=NULL;
    # EulerEdge 	* ee=NULL;#
    # unsigned int * 	levEdge=NULL;
    # unsigned int * 	entEdge=NULL;
    # unsigned int  	edgeCount=0;
    # unsigned int 	vertexCount=0;
    # unsigned int 	readCount=0;
    logging.info('Openinp %s' % (infile))
    buffer = Fasta(open(infile))

    readBuffer = [s.sequence for s in buffer]

    readCount = len(readBuffer)
    logging.info("Got %s reads." % (readCount))
    readLength = [len(seq) for seq in readBuffer]
    evList = [ ]
    eeList = [ ]
    levEdgeList = [ ]
    entEdgeList = [ ]
    edgeCountList = [ ]
    vertexCountList = [ ]

    if readCount > 0:
        if errorCorrection:
            readCount = doErrorCorrection ( readBuffer, readCount, ec_tuple_size, max_ec_pos )
        constructDebruijnGraph ( readBuffer, readCount, readLength,
                                 lmerLength, evList, eeList, levEdgeList, entEdgeList, edgeCountList, vertexCountList )
        # findEulerTour(evList, eeList, levEdgeList, entEdgeList, edgeCountList, vertexCountList, lmerLength, outfile)


if __name__ == '__main__':
    """
    """
    logging.config.fileConfig('logging.cfg')
    logger = logging.getLogger('eulercuda')
    logger.info('Program started')
    parser = argparse.ArgumentParser ( )

    parser.add_argument ( '-i', action = 'store', dest = 'input_filename',
                          help = 'Input Fie Name' )
    parser.add_argument ( '-o', action = 'store', dest = 'output_filename',
                          help = 'Output File Name' )
    parser.add_argument ( '-k', action = 'store', dest = 'k', type = int,
                          help = 'kmer size' )
    parser.add_argument ( '-d', action = 'store_true', default = False,
                          help = 'Use DDFS' )
    results = parser.parse_args ( )
    # Need to process commandline args. Probably just copy=paste from disco3
    if results.input_filename == '':
        fname = '../data/read_1.fq'
    else:
        fname = results.input_filename
    # readBuffer = read_fastq(fname)
    # assemble2(inputFileName, outputFileName, readLength, assemble,
    # lmerLength, coverage,errorCorrection, max_ec_pos,ec_tuple_size);

    # void assemble2(	const char * filename, 	//input filename
    # 		const char * output, 	//output filename
    # 		unsigned int readLength,	//readLength
    # 		bool assemble,
    # 		unsigned  int l,		//lmer length
    # 		unsigned int coverage,	//coverage M
    # 		bool errorCorrection,
    # 		unsigned int max_ec_pos,	//ec positions
    # 		unsigned int ec_tuple_size	//ec tuple size
    # 		){

    assemble2(results.input_filename, results.output_filename, 17, False, 20, 20)
