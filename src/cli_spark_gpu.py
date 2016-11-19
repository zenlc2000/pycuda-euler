from pyspark import SparkContext, SparkConf
import subprocess
import eulercuda as ec
import pycuda.driver
import pycuda.autoinit
import sys


complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

#raw_data = sc.textFile('hdfs://172.31.26.32:8020/genome/Ba10k.sim1.fq')

#data = raw_data.filter(lambda x: x[0] in ['A','C','G','T'])
#dataLength = len(data.take(1)[0])
#dataCount = data.count() // data.getNumPartitions()

#k = 17
#lmerLength = 18

def hail_mary(sc, path):
	import eulercuda.eulercuda as ec
#	import os
#	print (os.environ['PATH'])

	raw_data = sc.textFile('hdfs://172.31.26.32:8020/genome/sra_data.fastq',100)

	data = raw_data.filter(lambda x: len(x) > 0 and x[0] in ['A','C','G','T'])
	dataLength = len(data.take(1)[0])
	dataCount = data.count() // data.getNumPartitions()

	k = 17
	lmerLength = 18
	try:
		subprocess.call(["hdfs", "dfs", "-rm", "-r", "-f", path])
	except:
		pass
	hail_mary = data.mapPartitions(lambda x: ec.assemble2(k, buffer=x, readLength = dataLength,readCount=dataCount)).saveAsTextFile('hdfs://172.31.26.32/genome/sra_output')

if __name__ == '__main__':
	conf = SparkConf()
	conf.setAppName('GPU_Euler')
#	conf.setMaster('local[*]')
	sc = SparkContext(conf=conf)
	hail_mary(sc, '/genome/sra_output')
	sc.stop()
