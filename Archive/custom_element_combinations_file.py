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
import gc
import numpy as np
import pandas as pd


def create_combinations(file):
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    initial_path ='./training/pickles/standard and documentation/custom_elements/trial/'
    final_path = './training/pickles/standard and documentation/custom_element_combination_trial/'
    completed_file_path ='./training/pickles/standard and documentation/custom_elements_processed_trial/'
    custom = pd.read_pickle(initial_path+file, compression='gzip')
    custom = custom.drop_duplicates(subset=['category', 'element'])
    custom['element'] = custom['element'].str.replace(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ')
    total_rows = len(custom.index)
    logging.warning('Processing element : ' + file + 'Number of rows to combine: '+ str(total_rows))
    cat = []
    ele = []
    combined_df = pd.DataFrame(columns=['category', 'element'])
    logging.warning('creating combinations')
    k=1
    for key, data in custom.iterrows():
        words = data['element']#.split()
        logging.warning(words)
        words2 = words.replace('%', '%%').replace(' ', '%s')
        logging.warning('Number of words to combine: '+ str(len(words.split())))
        for i in itertools.product((' ', ''), repeat=words.count(' ')):
            ele.append(words2 % i)
            cat.append(data['category'])
        lst = zip(cat,ele)
        if len(lst) > 200000:
            del cat
            del ele
            combined_df = pd.DataFrame.from_records(lst,columns=['category','element'])
            del lst
            combined_df.to_pickle(final_path + os.path.splitext(file)[0] + str(k)+'.pickle', compression='gzip')
            combined_df.to_csv(final_path + os.path.splitext(file)[0] + str(k)+'.csv') 
            #del combined_df
            gc.collect()
            k+=1
    del cat
    del ele
    combined_df = pd.DataFrame.from_records(lst,columns=['category','element'])
    del lst
    combined_df.to_pickle(final_path + os.path.splitext(file)[0] + str(k)+'.pickle', compression='gzip')
    combined_df.to_csv(final_path + os.path.splitext(file)[0] + str(k)+'.csv') 
    del combined_df
    gc.collect()
    del custom
    del words
    del words2
    logging.warning('completed ' + file)
    os.rename(initial_path+file, completed_file_path+file)
    os.rename(initial_path+os.path.splitext(file)[0]+'.csv', completed_file_path+os.path.splitext(file)[0]+'.csv')
    return True


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    partitions = 1 #number of partitions to split dataframe
    cores = 6 #number of cores on your machine
    path ='./training/pickles/standard and documentation/custom_elements/trial/'
    combi_path = './training/pickles/standard and documentation/custom_element_combination_trial/'
    processed_file_path ='./training/pickles/standard and documentation/custom_elements_processed_trial/'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    pickle_files=[]
    for any_file in files:
        if any_file.endswith('.pickle'):
            if os.path.isfile(combi_path+any_file):
                os.rename(path+any_file, processed_file_path+any_file)
                os.rename(path+os.path.splitext(any_file)[0]+'.csv', processed_file_path+os.path.splitext(any_file)[0]+'.csv')
                logging.warning(any_file +' already processed.')
            else:
                df = pd.read_pickle(path+any_file, compression='gzip')
                rows = len(df.index)
                if rows > 0:
                    #if rows < 500:
                    pickle_files.insert(len(pickle_files),any_file)
                    # else:
                    #     continue
                else:
                    os.rename(path+any_file, processed_file_path+any_file)
                    os.rename(path+os.path.splitext(any_file)[0]+'.csv', processed_file_path+os.path.splitext(any_file)[0]+'.csv')
                del df
                gc.collect()
                del rows
                gc.collect()
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Pool(processes=cores, maxtasksperchild=1000)
    start = time.time()
    async_result = p.map_async(create_combinations, pickle_files)
    p.close()
    p.join()
    print("Complete")
    end = time.time()
    print('total time (s)= ' + str(end-start))
