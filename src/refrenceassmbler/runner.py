import referenceAssembler
import sys
from dask import delayed

#@delayed
def ingest(src):
    print(src)

#@delayed 
def run(src,k=1):
    referenceAssembler.runAssembler(k,src)



if __name__ == '__main__':
    ingest(sys.argv[1])
    run(sys.argv[1],1)