from pyspark import SparkConf, SparkContext
import time

def twin(km):
	complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    # return Seq.reverse_complement(km)
	return "".join(complement.get(base, base) for base in reversed(km))
        

def fw(km):
    for x in 'ACGT':
        yield km[1:]+x

def bw(km):
    for x in 'ACGT':
        yield x + km[:-1]

def contig_to_string(c):
    return c[0] + ''.join(x[-1] for x in c[1:])

def get_contig(d,km):
    '''
    Find kmer's contig.
    Return: the string, list of kmers in contig
    '''
    c_fw = get_contig_forward(d,km)

    c_bw = get_contig_forward(d,twin(km))

    if km in fw(c_fw[-1]):
        c = c_fw
    else:
        c = [twin(x) for x in c_bw[-1:0:-1]] + c_fw
    return contig_to_string(c),c


def get_contig_forward(d,km):
    c_fw = [km]

    while True:
        if sum(x in d for x in fw(c_fw[-1])) != 1:
            break

        cand = [x for x in fw(c_fw[-1]) if x in d][0]
        if cand == km or cand == twin(km):
            break # break out of cycles or mobius contigs
        if cand == twin(c_fw[-1]):
            break # break out of hairpins

        if sum(x in d for x in bw(cand)) != 1:
            break

        c_fw.append(cand)

    return c_fw

def all_contigs(k_tuples):
    d = dict(k_tuples)
    done = set()
    r = []
    for x in d:
        if x not in done:
            s,c = get_contig(d,x)
            for y in c:
                done.add(y)
                done.add(twin(y))
            r.append(s)
    return r


def main(sc):
	compliment = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
	raw_data = sc.textFile('hdfs://172.31.26.32:8020/genome/sra_data.fastq', 100)
	k = 21
	data = raw_data.filter(lambda x: len(x) >0 and x[0] in ['A','C','G','T'])
	fwd_list = data.flatMap(lambda x: [x[i:i+k] for i in range(len(x.rstrip())-k+1)])
	
	
	rev_comp = data.map(lambda x:''.join(reversed([compliment.get(base, base) for base in x])))
	
	rev_list = rev_comp.flatMap(lambda x: [x[i:i+k] for i in range(len(x.rstrip())-k+1)])
	kmer_list = fwd_list + rev_list
	emitter = kmer_list.map(lambda x: (x, 1))
	kmer_counts = emitter.reduceByKey(lambda x, y: x+y)
	
	t1 = time.time()
	t0 = time.clock()
	contigs = kmer_counts.mapPartitions(all_contigs)
	contigs.saveAsTextFile('hdfs://172.31.26.32:8020/genome/sra_out.txt')
	proc_time = time.clock() - t0
	wall_time = time.time() - t1

#	with open('output.out',w) as ofile:	
#		ofile.write(str(proc_time) + " process time.\n")
#		ofile.write(str(wall_time) + " wall time.\n")
#		ofile.write(str(len(contigs)) + " contigs fount.\n")


if __name__ =='__main__':
	conf = SparkConf()
#	conf.setMaster("yarn-cluster")
# 	conf.set('spark.driver.memory','8g')
#	conf.set('spark.executor.memory','1g')
#	conf.set('spark.executor.cores', '2')
#	conf.set('spark.executor.instances','12')
# 	conf.set('spark.shuffle.service.enabled','true') 
# 	conf.set('spark.dynamicAllocation.enabled','true')
#	conf.set('spark.dynamicAllocation.initialExecutors','8')
#	conf.set('spark.dynamicAllocation.minExecutors','4')
	conf.setAppName("PycudaEuler")
	sc = SparkContext(conf=conf)
	
	main(sc)

	sc.stop()
