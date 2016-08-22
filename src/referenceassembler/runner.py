import referenceAssembler
import sys
from dask import delayed

#@delayed
def ingest(src):
    print(src)

#@delayed(pure=True) 
def run(src,k=21):
    referenceAssembler.runAssembler(k,src)
    return 1


if __name__ == '__main__':
    print(sys.argv[1])
    #ingest(sys.argv[1])
    #v = run(sys.argv[1])