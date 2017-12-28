# pylint: disable=I0011
# pylint: disable=C0111
# pylint: disable=C0301
# pylint: disable=C0103

#import re
import logging
import os.path
import pickle
import pandas as pd
from input_dataset import sfp_category, soi_category, scf_category


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
cc = {}
cc['SFP'] = sfp_category
cc['SOI'] = soi_category
cc['SCF'] = scf_category
statements = ('SFP', 'SOI', 'SCF')


#logging.warning('   Started Processing of ' + cats[0] +'.') 
filename = './training/pickles/dataset/SFP/DebtCurrent.pickle'
final_file_path = './training/pickles/training_sets/SFP/DebtCurrent.pickle'

if os.path.isfile(filename):
    fulldataset = pd.read_pickle(filename, compression='gzip')
    dataset = fulldataset.drop_duplicates() 
    training_data = pd.DataFrame(columns=['category', 'element'])
    for key, data in dataset.iterrows():
    #for key, data in fulldataset.iterrows():
        element_file = './training/pickles/documentation/' +  (data['element']) +'.pickle'
        if os.path.isfile(element_file):
            df = pd.read_pickle(element_file, compression='gzip')
            df['category'] = data['category']
            training_data = training_data.append(df, ignore_index=True)
            del df
        else:
            logging.warning(element_file + ' does not exist.')
        logging.warning('       Completed Processing of ' + data['element'] +'.')

    #fulldataset=fulldataset.append(pd.concat([fulldataset['category'], fulldataset['element'].str.replace(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ')], axis=1, join_axes=[fulldataset.index]), ignore_index=True)
    fulldataset['element'] = fulldataset['element'].str.replace(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ')
    fulldataset = fulldataset.append(training_data, ignore_index=True)
    del training_data
    fulldataset.to_pickle(final_file_path, compression='gzip')
    logging.warning('       Pickled ' + final_file_path +'.')
    del fulldataset
    #del dataset
else:
    logging.warning(filename + ' dose not exist.')
#logging.warning('   Completed Processing of ' + cats[0] +'.')
#logging.warning('Completed Processing of ' + statements[i] +'.')