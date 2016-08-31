import os
import gc
import argparse
import encoder.pyencode as enc
import debruijn.pydebruijn as db
import gpuhash.pygpuhash as gh
import eulertour.pyeulertour as et
import numpy as np
from numpy import array
import logging
import logging.config
from fastareader.parse_fasta import Fasta
import pycuda.driver
from graph_tool.all import *

pycuda.driver.set_debugging(True)


def parse_fastq(filename):
    """
    Read fastq formatted <filename> and return a dictionary of
    read_name : read
    """
    file = open(filename)
    result = {}
    current_name = None
    for i, line in enumerate(file):
        if i % 4 == 0:
            current_name = line.rstrip('\n')
        if i % 4 == 1:
            result[current_name] = line.rstrip('\n')
        if i % 4 == 3:
            print('quality is ' + line.rstrip('\n'))
    return result


def read_fastq(filename):
    """
    Read fastq formatted <filename> and return a list of reads
    """
    with open(filename, "r") as infile:
        result = []
        for i, line in enumerate(infile):
            if i % 4 == 1:
                result.append(line.rstrip('\n'))
        return result


def doErrorCorrection(readBuffer, readCount, ec_tuple_size, max_ec_pos):
    return readCount


def readLmersKmersCuda(readBuffer, readLength, readCount, lmerLength, lmerKeys, lmerValues, lmerCount, kmerKeys,
                       kmerValues, kmerCount):
    """

    """
    logger = logging.getLogger('eulercuda.readLmersKmersCuda')
    logger.info("started.")
    kmerMap = {}
    lmerMap = {}

    # numpy type 'S' == Python-compatible string
    # numpy type 'Q' == C unsigned long long
    # numpu type 'I' == C unsigned int
    buffer = np.array(readBuffer).astype('S')
    nbr_values = buffer.size * buffer.dtype.itemsize
    d_lmers = np.zeros(buffer.size).astype('Q')
    d_pkmers = np.zeros_like(d_lmers)
    d_skmers = np.zeros_like(d_lmers)

    # CUDA_NUM_READS = 1024 * 32
    # if readCount < CUDA_NUM_READS:
    #     readToProcess = readCount
    # else:
    #     readToProcess = CUDA_NUM_READS
    CUDA_NUM_READS = 1024 * 32
    if readCount < CUDA_NUM_READS:
        readToProcess = readCount
    else:
        readToProcess = CUDA_NUM_READS
    # readToProcess = readCount
    kmerBitMask = 0

    # bufferSize = sum(sys.getsizeof(seq) for seq in readBuffer)
    # entriesCount = readLength * readCount
    # readLength = [len(seq) for seq in readBuffer]

    for _ in range(0, (lmerLength - 1) * 2):
        kmerBitMask = (kmerBitMask << 1) | 1
    logger.debug("kmerBitMask = %s" % kmerBitMask)

    readProcessed = 0
    # Originally a loop slicing readBuffer into chunks then process each chunk
    # Theoretically shouldn't have to do this on distrib. system.

    # while readProcessed < total_base_pairs:
    enc.encode_lmer_device(buffer, readCount, d_lmers, readLength, lmerLength)

    enc.compute_kmer_device(d_lmers, d_pkmers, d_skmers, kmerBitMask, readLength, readCount)
    h_lmersF = np.array(d_lmers)
    h_pkmersF = np.array(d_pkmers)
    h_skmersF = np.array(d_skmers)

    enc.compute_lmer_complement_device(buffer, readCount, d_lmers, readLength, lmerLength)
    enc.compute_kmer_device(d_lmers, d_pkmers, d_skmers, kmerBitMask, readLength, readCount)
    h_lmersR = np.array(d_lmers)
    h_pkmersR = np.array(d_pkmers)
    h_skmersR = np.array(d_skmers)

    lmerEmpty, kmerEmpty = 0, 0
    validLmerCount = readLength - lmerLength + 1
    # Here he fills the kmerMap and lmerMap with a nested for loop
    # for j in range(readToProcess):
    #     for i in range(validLmerCount):
    for index in range(readToProcess):
        # index = j * readLength + i
        kmerMap[h_pkmersF[index]] = 1
        kmerMap[h_skmersF[index]] = 1
        kmerMap[h_pkmersR[index]] = 1
        kmerMap[h_skmersR[index]] = 1

        if h_lmersF[index] == 0:
            lmerEmpty += 1
        else:
            if lmerMap.get(h_lmersF[index]) is None:
                lmerMap[h_lmersF[index]] = 1
            else:
                lmerMap[h_lmersF[index]] += 1
        if h_lmersR[index] == 0:
            lmerEmpty += 1
        else:
            if lmerMap.get(h_lmersR[index]) is None:
                lmerMap[h_lmersR[index]] = 1
            else:
                lmerMap[h_lmersR[index]] += 1
    # readProcessed += readToProcess
    # readToProcess -= readCount
    # if readCount < CUDA_NUM_READS:
    #     readToProcess = readCount
    # else:
    #     readToProcess = CUDA_NUM_READS
    #     # End of chunking loop

    kmerCount = len(kmerMap) + kmerEmpty
    # TODO: Log message with kmer count
    logger.info('kmer count = %d' % kmerCount)

    # kmerKeys = []
    # kmerValues = []
    #
    # I know, I know. I couldn't get enumerate to work.
    index = 0
    for k, v in kmerMap.items():
        kmerKeys.append(k)
        kmerValues.append(index)
        index += 1

    # original code has if below, but I don't see that it will ever be executed
    #     if (kmerEmpty > 0){
    #     ( * kmerKeys)[index]=0;
    #     ( * kmerValues)[index]=index;
    #     }

    lmerCount = len(lmerMap) + lmerEmpty
    #
    lmerKeys = [key for key in lmerMap.keys()]
    lmerValues = [value for value in lmerMap.values()]

    # lmerKeys = t_lmerKeys.
    if lmerEmpty > 0:
        lmerKeys[len(lmerMap) - 1] = 0
        lmerValues[len(lmerMap) - 1] = lmerEmpty

    return [lmerCount, kmerCount, lmerKeys, lmerValues, kmerKeys, kmerValues]


def constructDebruijnGraph(readBuffer, readCount, readLength, lmerLength, evList, eeList, levEdgeList, entEdgeList):
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
    logger = logging.getLogger('eulercuda.constructDebruijnGraph')
    logger.info('Begin constructing deBruijn graph')

    h_lmerKeys = []
    h_lmerValues = []

    lmerCount = 0
    h_kmerKeys = []
    h_kmerValues = []
    d_kmerKeys = []
    d_kmerValues = []
    kmerCount = 0
    d_TK = []
    d_TV = []
    tableLength = 0
    bucketCount = 0
    d_bucketSize = [[], []]

    coverage = 20
    d_ev = []
    d_ee = []
    d_levEdge = []
    d_entEdge = []

    # May need to return unpacked tuple of integer variables
    vals = readLmersKmersCuda(readBuffer, readLength, readCount, lmerLength, h_lmerKeys, h_lmerValues,
                              lmerCount, h_kmerKeys, h_kmerValues, kmerCount)
    lmerCount = vals[0]
    kmerCount = vals[1]
    h_lmerKeys = vals[2]
    h_lmerValues = vals[3]
    h_kmerKeys = vals[4]
    h_kmerValues = vals[5]
    # initDevice()
    # setStatItem(NM_LMER_COUNT, lmerCount);
    # setStatItem(NM_KMER_COUNT, kmerCount);


    logger.debug('lmerCount = %d' % lmerCount)
    logger.debug('projected kmer count: %s, actual: %s' % ((readCount * (readLength - lmerLength)), kmerCount))
    # TODO: return d_TK,  d_TV, tableLength,  d_bucketSize,  bucketCount)
    # TODO: Test pygpuhash
    # =======> pygpuhash.create_hash_table(d_kmerKeys, d_kmerValues, kmerCount)
    results = gh.create_hash_table_device(h_kmerKeys, h_kmerValues, kmerCount, d_TK, d_TV, tableLength,
                                          d_bucketSize, bucketCount)

    tableLength = results[0]
    d_bucketSize = results[1]
    bucketCount = results[2]
    d_TK = results[3]
    d_TV = results[4]
    # TODO:
    #       unsigned int * ecount,
    # 		KEY_PTR d_lmerKeys, //in lmer keys
    # 		VALUE_PTR d_lmerValues, //in lmer values
    # 		unsigned int lmerCount, //in total lmers
    # 		KEY_PTR d_kmerKeys, //in
    # 		unsigned long kmerCount, //in  total kmers
    # 		unsigned int l, //in k
    # 		KEY_PTR d_TK,
    # 		VALUE_PTR d_TV,
    # 		unsigned int * d_bucketSeed,
    # 		unsigned int bucketCount,
    # 		EulerVertex ** d_ev, //out
    # 		unsigned int ** d_l, //out
    # 		unsigned int ** d_e, //out
    # 		EulerEdge ** d_ee //out

    d_ev, d_ee, d_levEdge, d_entEdge, kmerCount, edgeCount = db.construct_debruijn_graph_device(h_lmerKeys, h_lmerValues, lmerCount,
                                       h_kmerKeys, kmerCount, lmerLength, d_TK, d_TV, d_bucketSize, bucketCount,
                                       d_ev, d_levEdge, d_entEdge, d_ee, readLength, readCount)
    logger.info("Finished. Leaving")
    return d_ev, d_ee, d_levEdge, d_entEdge, kmerCount, edgeCount


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


def findSpanningTree(cg_edge, cg_edgecount, cg_vertexcount,):
    """
    :param cg_edge:
    :param cg_edgecount:
    :param cg_vertexcount:
    :param tree:
    :return:
    """
    logger = logging.getLogger('eulercuda.findSpanningTree')
    logger.info('Begin spanning tree search')

    # int * with length=cg_edgecount
    # with each value initially set to 1
    weights = [1] * cg_edgecount

    g = Graph(directed=False)
    g.add_vertex(cg_vertexcount)
 
    # edge_index 
    indexMap =  g.new_edge_property('int')
    weightMap = g.new_edge_property('int')

    # j = 0
    for j, edge in enumerate(cg_edge):
        e = g.add_edge(edge['c1'], edge['c2'])
        weightMap[e] = weights[j]
        indexMap[e] = j
        # j += 1
 
    index = g.new_edge_property('int')
 
    # when called without the root argument this
    # uses kruskal's algorithm
    treeMap = graph_tool.topology.min_spanning_tree(g)
  
    # build uint ** for passing to cuda  
    tree = np.zeros((cg_edgecount,2),dtype=np.uintc)

    for i,e in enumerate(treeMap):
        tree[i] = np.asarray(e)
     
    return tree

def dna_translate(i):
    bases = {0:'A', 1:'C', 2:'G', 3:'T'}
    if i < 4:
        return bases[i]
    else:
        return '.'

def getString(length, value):
    kmer = [0] * length
    currentValue = int(value)
    for i in range(1, length + 1):
        kmer[length - i] = dna_translate(currentValue % 4)
        currentValue //= 4
    return ''.join(kmer)


def generatePartialContig(outfile, d_ev, vcount, d_ee, ecount, l):
    logger = logging.getLogger(__name__)
    logger.info("Starting")
    d_contigStart = np.ones(ecount, dtype=np.uintc)
    d_contigStart = et.identify_contig_start(d_ee, d_contigStart, ecount)
    h_contigStart = np.copy(d_contigStart)

    h_ev = np.zeros(vcount, dtype=[('vid', np.uintc), ('n1', np.uintc), ('n2', np.uintc)])
    h_ee = np.zeros(ecount, dtype=[('ceid', np.uintc), ('e1', np.uintc), ('e2', np.uintc), ('c1', np.uintc), ('c2', np.uintc)])
    buffer = []
    h_visited = np.zeros(ecount, dtype=np.uintc)
    h_ev = np.copy(d_ev)
    h_ee = np.copy(d_ee)

    count = 0
    edgeCount = 0
    next = None

    with open(outfile, 'w') as ofile:
        for i in range(ecount):
            if h_contigStart[i] != 0 and h_visited[i] == 0:
                ofile.write('>%u\n' % count)
                logger.debug('>%u\n' % count)
                count += 1
                buffer.append(getString(l -1, h_ev[h_ee[i]['v1']]['vid']))
                # ofile.write(''.join(buffer))
                # logger.debug(''.join(buffer))
                next = i
                while h_ee[next]['s'] < ecount and h_visited[h_ee[next]['s']] == 0:
                    h_visited[next] = 1
                    next = h_ee[next]['s']
                    buffer.append(getString(l - 1, h_ev[h_ee[next]['s']]['vid'] ))
                    # ofile.write('%s' % ''.join(buffer) + str(l - 2))
                    # logger.debug('%s' % ''.join(buffer) + str(l - 2))
                    edgeCount += 1
                if h_visited[next] == 0: # for circular paths
                    buffer.append(getString(l - 1, h_ev[h_ee[next]['v2']]['vid']))
                    # ofile.write('%s' % buffer + str(l - 2))
                    # logger.debug('%s' % buffer + str(l - 2))
                    h_visited[next] = 1
                    edgeCount += 1
                ofile.write(''.join(buffer) + '\n\n')
                buffer = []
        for i in range(ecount):
            if h_visited[i] == 0:
                ofile.write('>%u\n' % count)
                logger.debug('>%u\n' % count)
                count += 1
                buffer.append(getString(l -1, h_ev[h_ee[i]['v1']]['vid']))
                # ofile.write(''.join(buffer))
                # logger.debug(''.join(buffer))
                next = i
                while h_ee[next]['s'] < ecount and h_visited[h_ee[next]['s']] == 0:
                    h_visited[next] = 1
                    next = h_ee[next]['s']
                    buffer.append(getString(l - 1, h_ev[h_ee[next]['s']]['vid'] ))
                    # ofile.write('%s' % ''.join(buffer) + str(l - 2))
                    # logger.debug('%s' % ''.join(buffer) + str(l - 2))
                    edgeCount += 1
                if h_visited[next] == 0: # for circular paths
                    buffer.append(getString(l - 1, h_ev[h_ee[next]['v2']]['vid']))
                    # ofile.write('%s' % buffer + str(l - 2))
                    # logger.debug('%s' % buffer + str(l - 2))
                    h_visited[next] = 1
                    edgeCount += 1
                ofile.write(''.join(buffer) + '\n\n')
                buffer = []
    logger.info("Finished")







def findEulerTour(d_ev, d_ee, d_levEdge, d_entEdge, edgeCountList, vertexCount, lmerLength, outfile):
    """

    :param ev_dict:
    :param ee_dict:
    :param levEdgeList:
    :param entEdgeList:
    :param edgeCount:
    :param vertexCount: kmerCount from debruijn graph
    :param lmerLength:
    :param outfile:
    :return:
    """
    logger = logging.getLogger("eulercuda.findEulerTour")
    logger.info("Started")
    d_cg_edge = {} # CircuitEdge struct
    cg_edgeCount = 0 # np.uintc
    cg_vertexCount = 0 # np.uintc

    cg_edge, cg_edgeCount, cg_vertexCount = et.findEulerDevice(
        d_ev, d_levEdge, d_entEdge, vertexCount, d_ee, edgeCountList, d_cg_edge,
        cg_edgeCount, cg_vertexCount, kmerLength)
    vertexCount, edgeCount = cg_vertexCount, cg_edgeCount
    if cg_edgeCount > 0:
        tree = findSpanningTree(cg_edge, cg_edgeCount, cg_vertexCount)
        d_ee = et.executeSwipeDevice(d_ev, d_entEdge, vertexCount, d_ee, edgeCount, cg_edge, cg_edgeCount, tree, len(tree))
        generatePartialContig(outfile, d_ev, vertexCount, d_ee, edgeCount, lmerLength)

    logger.info("finished")

def read_fasta(infilename):
    sequence = []
    with open(infilename, 'r') as infile:
        for line in infile:
            if line[0] != '>':
                sequence.append(line.strip().encode('ascii'))
    return sequence


def assemble2(infile, outfile, lmerLength, errorCorrection, max_ec_pos, ec_tuple_size):
    """
    Do the assemble
    """
    # TODO: figure out logging
    # TODO: Unit testing
    logger = logging.getLogger(__name__)
    # for performance reasons, may want to make these Numpy arrays

    # char * 		readBuffer=NULL;
    # EulerVertex * 	ev=NULL;
    # EulerEdge 	* ee=NULL;#
    # unsigned int * 	levEdge=NULL;
    # unsigned int * 	entEdge=NULL;
    # unsigned int  	edgeCount=0;
    # unsigned int 	vertexCount=0;
    # unsigned int 	readCount=0;
    logger.info('Opening %s' % infile)

    extension = infile.split('.')[-1]
    # buffer = Fasta(open(infile))
    # readBuffer = [s.sequence for s in buffer if len(s.sequence) >= lmerLength]
    # readBuffer = list(''.join(readBuffer))
    if extension in ['fa', 'fasta', 'fsa']:
        buffer = read_fasta(infile)
    elif extension in ['fq', 'fastq']:
        readBuffer = read_fastq(infile)

    readLength = len(buffer[0])
    # cull out the shorties
    buffer = [r for r in buffer if len(r) == readLength]
    readCount = len(buffer)
    logger.info("Got %s reads." % (readCount))

    # total_base_pairs = readCount * readLength
    evList = []
    eeList = []
    edgeCount = 0
    levEdgeList = []
    entEdgeList = []
    edgeCount = 0
    vertexCount = 0

    if readCount > 0:
        if errorCorrection:
            readCount = doErrorCorrection(readBuffer, ec_tuple_size, max_ec_pos)
        eeList, evList, levEdgeList, entEdgeList, vertexCount, edgeCount = constructDebruijnGraph(buffer, readCount, readLength,
                               lmerLength, evList, eeList, levEdgeList, entEdgeList)
        findEulerTour(evList, eeList, levEdgeList, entEdgeList, edgeCount, vertexCount, lmerLength, outfile)


if __name__ == '__main__':
    """
    """
    print(os.getpid())
    # input("=> ")
    logging.config.fileConfig('logging.cfg')
    logger = logging.getLogger('eulercuda')
    logger.info('Program started')
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', action='store', dest='input_filename', help='Input Fie Name')
    parser.add_argument('-o', action='store', dest='output_filename', help='Output File Name')
    parser.add_argument('-k', action='store', dest='k', type=int, help='kmer size')
    parser.add_argument('-d', action='store_true', dest='debug', default=False)
    results = parser.parse_args()

    if results.input_filename == '':
        fname = '../data/Ecoli_raw.fasta'
    else:
        fname = results.input_filename

    if results.k > 0:
        kmerLength = results.k
    else:
        kmerLength = 21
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

    assemble2(results.input_filename, results.output_filename, kmerLength, False, 20, 20)
