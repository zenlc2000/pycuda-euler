from dask import delayed


srcfile = ""

def loadData(src):
    print("loading data")
    if src.startswith("s3://"):
        print("data from s3")
    elif src.startswith("file://"):
        print("data from file")
    else:
        print("unknown data source")
        


dsk = {'load-fastq': (loadData, srcfile),
       'clean-1': ('clean', 'load-1'),
       'clean-2': ('clean', 'load-2'),
       'clean-3': ('clean', 'load-3'),
       'analyze': ('analyze', ['clean-%d' % i for i in [1, 2, 3]]),
       'store'  : ('store', 'analyze')}