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

for i in range(0, 3):
    logging.warning('Started Processing of ' + statements[i] +'.') 
    for key, cats in cc[statements[i]].items():
        logging.warning('   Started Processing of ' + cats[0] +'.') 
        filename = './training/pickles/standard and documentation/' + statements[i] +'/'+ cats[0] +'.pickle'
        final_file_path = './training/pickles/standard and documentation/training_sets/' + statements[i] +'/' + cats[0] +'.pickle'
        csv_file_path = './training/pickles/standard and documentation/training_sets/' + statements[i] +'/' + cats[0] +'.csv'
        if os.path.isfile(filename):
            fulldataset = pd.read_pickle(filename, compression='gzip')
            dataset = fulldataset.drop_duplicates()
            dataset = dataset[~dataset['element'].isin(dataset['category'])].reset_index() 
            training_data = pd.DataFrame(columns=['category', 'element'])
           
            # Reading Documentation start
            #for key, data in fulldataset.iterrows():
            for key, data in dataset.iterrows():
                element_file = './training/pickles/standard and documentation/documentation/' +  (data['element']) +'.pickle'
                if os.path.isfile(element_file):
                    df = pd.read_pickle(element_file, compression='gzip')
                    df['element'] = df['category'].str.replace(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ').str.cat(df['element'], sep=' ')
                    df['category'] = data['category']
                    training_data = training_data.append(df, ignore_index=True)
                    del df
                else:
                    logging.warning(element_file + ' does not exist.')
                logging.warning('       Completed Processing of ' + data['element'] +'.')
            # # Reading Documentation end
            fulldataset=fulldataset.append(pd.concat([fulldataset['category'], fulldataset['element'].str.replace(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ')], axis=1, join_axes=[fulldataset.index]), ignore_index=True)
            fulldataset=fulldataset.append(pd.concat([fulldataset['category'], fulldataset['element'].str.lower()], axis=1, join_axes=[fulldataset.index]), ignore_index=True)
            
            #fulldataset['element'] = fulldataset['element'].str.replace(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ')]

            # Appending Documentation start
            fulldataset = fulldataset.append(training_data, ignore_index=True)
            # Appending Documentation end
            del training_data
            fulldataset.to_pickle(final_file_path, compression='gzip')
            fulldataset.to_csv(csv_file_path)
            logging.warning('       Pickled ' + final_file_path +'.')
            del fulldataset
            #del dataset
        else:
            logging.warning(filename + ' dose not exist.')
        logging.warning('   Completed Processing of ' + cats[0] +'.')
    logging.warning('Completed Processing of ' + statements[i] +'.')