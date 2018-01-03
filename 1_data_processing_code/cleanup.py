import pickle
import pandas as pd
import logging
import os
import itertools
from input_dataset import sfp_category, soi_category, scf_category


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
cc = {}
cc['SFP'] = sfp_category
cc['SOI'] = soi_category
cc['SCF'] = scf_category
statements = ('SFP', 'SOI', 'SCF')
path ='./training/pickles/standard and documentation/'
for i in range(0, 3):
    logging.warning('Started Processing of ' + statements[i] +'.') 
    for key, cats in cc[statements[i]].items():
        logging.warning('   Started Processing of ' + cats[0] +'.') 
        filename = path + statements[i] +'/'+ cats[0] +'.pickle'
        final_file_path = path + 'training_sets/' + statements[i] +'/' + cats[0] +'.pickle'
        csv_file_path = path + 'training_sets/' + statements[i] +'/' + cats[0] +'.csv'
        if os.path.isfile(filename):
            fulldataset = pd.read_pickle(filename, compression='gzip')
            dataset = pd.DataFrame(columns=['category', 'element'])
            dataset['element'] = fulldataset['element'].str.replace(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ')
            dataset['category'] = fulldataset['category']
            dataset = dataset.drop_duplicates()
            training_data = pd.DataFrame(columns=['category', 'element'])
            
            for key1, data in dataset.iterrows():
                element=data['element']
                words = element.split()
                k=0
                for t in itertools.product(range(len('01')), repeat=len(words)-1):
                    df = pd.DataFrame(columns=['category', 'element'])
                    df.loc[k] = [data['category'],''.join([words[j]+t[j]*' ' for j in range(len(t))])+words[-1]]
                    k+=1
                    training_data = training_data.append(df, ignore_index=True)
                    del df
                logging.warning('  Completed processing ' + element)
            fulldataset = fulldataset.append(training_data, ignore_index=True)
            fulldataset = fulldataset.append(pd.concat([fulldataset['category'], fulldataset['element'].str.lower()], axis=1, join_axes=[fulldataset.index]), ignore_index=True)
            del training_data
            fulldataset.to_pickle(final_file_path, compression='gzip')
            fulldataset.to_csv(csv_file_path)
            logging.warning('       Pickled ' + final_file_path +'.')
            del fulldataset

                    



# logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
# cc = {}
# cc['SFP'] = sfp_category
# cc['SOI'] = soi_category
# cc['SCF'] = scf_category
# statements = ('SFP', 'SOI', 'SCF')

# for i in range(0, 3):
#     logging.warning('Started Processing of ' + statements[i] +'.') 
#     for key, cats in cc[statements[i]].items():
#         final_file_path = './training/pickles/standard and documentation/' + statements[i] +'/' + cats[0] +'.pickle'
#         csv_file_path = './training/pickles/standard and documentation/' + statements[i] +'/' + cats[0] +'.csv'
#         if os.path.isfile(final_file_path):
#             df = pd.read_pickle(final_file_path, compression='gzip')
#             df = df[~df['element'].isin(df['category'])].reset_index()
#             if 'index' in df.columns:
#                 df = df.drop('index', 1)
#             df.to_pickle(final_file_path, compression='gzip')
#             df.to_csv(csv_file_path)
#             logging.warning('   Cleaned ' + cats[0] +'.') 
#         else:
#             logging.warning('   File ' + cats[0] +'.pickle does not exist.') 





