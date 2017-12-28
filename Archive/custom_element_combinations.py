# pylint: disable=I0011
# pylint: disable=C0111
# pylint: disable=C0301
# pylint: disable=C0103
import logging
import psutil
import os
import math
import numpy as np
from os import listdir
from os.path import isfile, join
import pickle
import pandas as pd
import itertools
import multiprocessing


def create_combinations(custom):
    combined_df = pd.DataFrame(columns=['category', 'element'])
    logging.warning('creating combinations')
    for key, data in custom.iterrows():
        words = data['element']#.split()
        logging.warning(words)
        words2 = words.replace('%', '%%').replace(' ', '%s')
        logging.warning('Number of words to combine: '+ str(len(words.split())))
        k=0
        combi_data = pd.DataFrame(columns=['category','element'])
        for i in itertools.product((' ', ''), repeat=words.count(' ')):
            df1 = pd.DataFrame(columns=['category','element'])
            df1.loc[k,'element']= (words2 % i)
            df1.loc[k,'category'] = data['category']
            combi_data = combi_data.append(df1, ignore_index=True)
            k+=1
            del df1
        combined_df = combined_df.append(combi_data, ignore_index=True) 
        del combi_data
 
##########################################
        # words = data['element'].split()
        # logging.warning('Number of words to combine: '+ str(len(words)))
        # k=0
        # combi_data = pd.DataFrame(columns=['category','element'])
        # for t in itertools.product(range(len('01')), repeat=len(words)-1):
        #     df1 = pd.DataFrame(columns=['category','element'])
        #     df1.loc[k,'element'] =''.join([words[j]+t[j]*' ' for j in range(len(t))])+words[-1]
        #     df1.loc[k,'category'] = data['category']
        #     combi_data = combi_data.append(df1, ignore_index=True)
        #     del df1
        # combined_df = combined_df.append(combi_data, ignore_index=True)   
        #####################################
    logging.warning('sending partial data')
    return combined_df


def parallelize_dataframe(df, func, num_cores, num_partitions ):
    logging.warning('splitting data')
    df_split = np.array_split(df, num_partitions)
    # all_cpus = list(range(psutil.cpu_count()))
    # p = psutil.Process()
    # p.cpu_affinity(all_cpus)
    pool = multiprocessing.Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    logging.warning('mearging all dataframes')
    pool.close()
    pool.join()
    #pool.close()
    
    return df
    
    


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    partitions = 1 #number of partitions to split dataframe
    cores = 4 #number of cores on your machine
    path='./to_process/'
    combi_path = './processed/'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for file in files:
        if file.endswith('.pickle'):
            if os.path.isfile(combi_path+file):
                logging.warning(file +' already processed.')
            else:
                custom_elements = pd.read_pickle(path+file,compression='gzip')
                custom_elements = custom_elements.drop_duplicates(subset=['category','element'])
                custom_elements['element'] = custom_elements['element'].str.replace(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ')
                total_rows=len(custom_elements.index)
                logging.warning('Processing element : ' + file)
                logging.warning('Number of rows to combine: '+ str(total_rows))
                if total_rows > cores:
                    partitions = math.floor(total_rows/cores)
                logging.warning('Number of partitions : ' + str(partitions))
                if total_rows > 0:
                    # df_final=create_combinations(custom_elements)
                    df_final = parallelize_dataframe(custom_elements, create_combinations, cores, partitions )
                    df_final.to_pickle(combi_path + file,compression='gzip')
                    df_final.to_csv(combi_path + os.path.splitext(file)[0]+'.csv') 
                    del df_final
                    del custom_elements
                    logging.warning('completed ' + file)
                else:
                    logging.warning('No rows to process')
                partitions = 1

                # split_df = pd.DataFrame(columns=['category', 'element'])
                # print('started' + file )
                # for key, data in custom.iterrows():
                #     split_df = split_df.append(split_data(data),ignore_index=True)
                
                






