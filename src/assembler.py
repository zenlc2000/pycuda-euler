
import os
import argparse
import logging
import subprocess


class Command:
    def execute(self): pass


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


"""
Do we need to set the kmer size?
"""

class AssemblerApp:
    
    def __init__(self):
        self.version = "1.0.0"
        self.parser = argparse.ArgumentParser()
        parser.add_argument('-i', action='store', dest='input_filename',
                        help='Input File Name')
        parser.add_argument('-o', action='store', dest='output_filename',
                        help='Output File Name')
        parser.add_argument('-k', action='store', dest='k', type=int,
                        help='kmer size')
        parser.add_argument('-d', action='store_true', default=False,
                        help='Use DDFS')
        results = parser.parse_args()

    @staticmethod
    def run(self,args):
        logging.info("Beginning Assembler run")


if __name__ == "__main__":
    AssemblerApp.run()
