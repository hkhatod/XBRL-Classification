# pylint: disable=I0011
# pylint: disable=C0111
# pylint: disable=C0301
# pylint: disable=C0103
# pylint: disable=W0612
# pylint: disable=W0611
import logging
import os
from os import listdir
from os.path import isfile, join
import math
import pickle
import itertools
import multiprocessing
import time
import psutil
import numpy as np
import pandas as pd


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    partitions = 1 #number of partitions to split dataframe
    cores = 2 #number of cores on your machine
    path ='./training/pickles/standard and documentation/custom_elements_processed/'
    combi_path = './training/pickles/standard and documentation/completed/'
    processed_file_path ='./training/pickles/standard and documentation/combination2/'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    pickle_files=[]
    for any_file in files:
        if any_file.endswith('.pickle'):
            if os.path.isfile(path+any_file):
                print(any_file)
                custom = pd.read_pickle(path+any_file, compression='gzip')
                if 'documentation' in custom.columns:
                    os.rename(path+any_file, combi_path+any_file)
                    os.rename(path+os.path.splitext(any_file)[0]+'.csv', combi_path+os.path.splitext(any_file)[0]+'.csv')
                else:
                    os.rename(path+any_file, processed_file_path+any_file)
                    os.rename(path+os.path.splitext(any_file)[0]+'.csv', processed_file_path+os.path.splitext(any_file)[0]+'.csv')
                