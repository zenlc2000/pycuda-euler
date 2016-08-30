import referenceAssembler
import sys
from dask import delayed, multiprocessing, threaded
from distributed import Executor

@delayed
def ingest(src):
    print(src)

@delayed
def run(src,k=21):
    referenceAssembler.runAssembler(k,src)
    return 1


if __name__ == '__main__':
    e = Executor('127.0.0.1:8786')
    print(sys.argv[1])
    #ingest(sys.argv[1])
    e.compute(run(sys.argv[1]))