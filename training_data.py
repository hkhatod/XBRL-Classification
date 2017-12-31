''' This programs aggregates training data from all elements under the parent element.
Aggregation starts with the parent file.
This file contains two columns: category and element.

1. Base elements            :   This data comes from parent file. 
                                This file contains two coluns - category and element.
2. Element documentations   :   This file contains two coluns - category and element.
                                The category column is ignore and the immidiate parent name is used as category.
3. custom elements          :   This file contains two coluns - category and element.
4. Base elements            :   This file containts n-grams of base elements.
                                This file contains one column -  element.
5. Custom elements n-grams  :   This file contains two coluns - category and element.
6. Retraining               :   To be implemented.
                                This data is the reclassified data which will be used to further refine
                                the model.
                                This file contains two coluns - category and element. '''
# pylint: disable=I0011
# pylint: disable=C0111
# pylint: disable=C0301
# pylint: disable=C0103
import logging
import os
import sys
import gc
import json
import pandas as pd
from input_dataset import sfp_category, soi_category, scf_category

def main():
    training_config = './code/training_config.json'
    params = json.loads(open(training_config).read())
    #process_element = sys.argv[1] replaced by  params['classify_element']
    
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    cc = {}
    cc['SFP'] = sfp_category
    cc['SOI'] = soi_category
    cc['SCF'] = scf_category
    statements = ('SFP', 'SOI', 'SCF')
    exclude = ('OtherAssetsCurrent',)

    # for i in range(0, 3):
    # logging.warning('Started Processing of ' + statements[i] +'.')
    #     for key, cats in cc[statements[i]].items():
    #         logging.warning('   Started Processing of ' + cats[0] +'.')
    # """Delete i when implementing for full program"""
    i = 0
    path = './training/pickles/standard and documentation/'
    #path = './training/pickles/standard and documentation/'
    logging.warning('   Started Processing of  {}'.format(params['classify_element']))
    parent = path + statements[i] +'/'+ params['classify_element'] +'.pickle'
    logging.warning('   parent: ' + parent)
    dest_dir = path + 'training_sets/' + statements[i] +'/' + params['classify_element']+'/'
    '''i is a folder increment varaiable. Its also used to update the name of tsv file.'''
    fld_spx = 1
    
    if os.path.exists(dest_dir):
        while os.path.exists(path + 'training_sets/' + statements[i] +'/' + params['classify_element']  + str(fld_spx) + '/'):
            
            fld_spx += 1
        dest_dir = path + 'training_sets/' + statements[i] +'/' + params['classify_element']+ str(fld_spx) + '/'
            
    os.makedirs(dest_dir)
    dest_pickle_file = dest_dir + params['classify_element'] +'.pickle'
    dest_csv_file = dest_dir + params['classify_element'] +'.csv'
    
    if os.path.isfile(parent): # Check if the parent file exist. If not, go to the next one
        fulldataset = pd.read_pickle(parent, compression='gzip') # load parent dataset
        dataset = pd.DataFrame(columns=['category', 'element'])
        '''
        Remove child with that dont have any further children.
        This dataset will be used to look up files for additional data aggregation.
        Drop duplicates so that we load each file only once.
        '''
        #dataset = fulldataset[~fulldataset['element'].isin(fulldataset['category'])].reset_index()
        #dataset['category'] = fulldataset['category']
        dataset = fulldataset
        dataset = dataset.drop_duplicates(subset=['category', 'element'])
        dataset.to_csv(dest_dir + params['classify_element']+'_dataset' + '.csv')
        fulldataset['element_name'] = fulldataset['element']
        '''
        Uncomment below line if not loading base element combinations.
        '''
        # fulldataset['element'] = fulldataset['element'].str.replace(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ')
        rows_to_process = len(dataset)
        '''
        iterate through each row in dataset and load:
        - documentation
        - element combination or ngrams
        - custom element combination or ngrams
        '''
        for key, data in dataset.iterrows():
            #logging.warning(key -1 +' processed out of ' + rows_to_process)
            '''
            Loading cleaned documentations. comment the IF BLOCK below if you want to exclude document.
            '''
            if params['documentation']:
                '''
                Define path for cleaned documentations. comment line below if you want to exclude document.
                '''
                doc_file = path + 'cleaned_docs/' +  (data['element']) +'.pickle'
                if os.path.isfile(doc_file) and data['element'] not in exclude:
                    dof = pd.read_pickle(doc_file, compression='gzip')
                    dof['element_name'] = dof['category']
                    if data['category'] == params['classify_element']:# replace params['classify_element'] with cat[0]
                        dof['category'] = data['element']
                    else:
                        dof['category'] = data['category']
                    fulldataset = pd.concat([fulldataset, dof], ignore_index=True)
                    del dof
                    gc.collect()
                    logging.warning('       Completed loading documentation of ' + data['element'] +'.')
                else:
                    logging.warning(doc_file + ' does not exist.')
            '''
            Loading base elements combinations or ngrams. comment the IF BLOCK below if you want to exclude base element combinations.
            '''
            if params['standard_ngrams']:
                combi_file = path + 'element_combination/'+ (data['element']) +'.pickle'
                if os.path.isfile(combi_file):
                    df_c = pd.read_pickle(combi_file, compression='gzip')
                    df_c['element_name'] = data['element']
                    if data['category'] == params['classify_element']: # replace params['classify_element'] with cat[0]
                        df_c['category'] = data['element']
                    else:
                        df_c['category'] = data['category']
                    fulldataset = pd.concat([fulldataset, df_c], ignore_index=True)
                    del df_c
                    gc.collect()
                    logging.warning('       Completed loading combinations of ' + data['element'] +'.')
                else:
                    logging.warning(combi_file + ' does not exist.')            

           
            if params['custom_elements']:
                cust_file = path + 'custom_elements_processed/'+ (data['element']) +'.pickle'
                cust_data = pd.DataFrame(columns=['category', 'element'])
                if (os.path.isfile(cust_file)) and (params['classify_element'] != data['element'] and data['element'] not in exclude): #(cats[0]!=data['element']):
                    df_ce = pd.read_pickle(cust_file, compression='gzip')
                    cust_data['element'] = df_ce['element'].str.replace(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ')
                    if data['category'] == params['classify_element']: # replace params['classify_element'] with cat[0]
                        cust_data['category'] = data['element']
                    else:
                        cust_data['category'] = data['category']
                        cust_data['element_name'] = data['element'] 
                    fulldataset = pd.concat([fulldataset, cust_data], ignore_index=True)
                    del df_ce
                    del cust_data
                    gc.collect()
                    logging.warning('       Completed loading custom elements of ' + data['element'] +'.')
                else:
                    logging.warning(cust_file + ' does not exist.')


            if params['custom_documentation']:
                cust_file = path + 'custom_documentation/'+ (data['element']) +'.pickle'
                cust_data = pd.DataFrame(columns=['category', 'element'])
                if (os.path.isfile(cust_file)) and (params['classify_element'] != data['element'] and data['element'] not in exclude): #(cats[0]!=data['element']):
                    df_ce = pd.read_pickle(cust_file, compression='gzip')
                    cust_data['element'] = df_ce['element']
                    if data['category'] == params['classify_element']: # replace params['classify_element'] with cat[0]
                        cust_data['category'] = data['element']
                    else:
                        cust_data['category'] = data['category']
                        cust_data['element_name'] = data['element'] 
                    fulldataset = pd.concat([fulldataset, cust_data], ignore_index=True)
                    del df_ce
                    del cust_data
                    gc.collect()
                    logging.warning('       Completed loading custom elements of ' + data['element'] +'.')
                else:
                    logging.warning(cust_file + ' does not exist.')

            if params['custom_ngrams']:
                cust_combi_file = path + 'custom_element_combination/'+ (data['element']) +'.pickle'
                cust_combi_data = pd.DataFrame(columns=['category', 'element'])
                if (os.path.isfile(cust_combi_file)) and (params['classify_element'] != data['element'] and data['element'] not in exclude): #(cats[0]!=data['element']):
                    df_cc = pd.read_pickle(cust_combi_file, compression='gzip')
                    cust_combi_data['element'] = df_cc['element']
                    if data['category'] == params['classify_element']: # replace params['classify_element'] with cat[0]
                        cust_combi_data['category'] = data['element']
                    else:
                        cust_combi_data['category'] = data['category']
                        cust_combi_data['element_name'] = data['element'] 
                    fulldataset = pd.concat([fulldataset, cust_combi_data], ignore_index=True)
                    del df_cc
                    del cust_combi_data
                    gc.collect()
                    logging.warning('       Completed loading custom combinations of ' + data['element'] +'.')
                else:
                    logging.warning(combi_file + ' does not exist.')
            
        #fulldataset = fulldataset.append(pd.concat([fulldataset['category'], fulldataset['element'].str.lower()], axis=1, join_axes=[fulldataset.index]), ignore_index=True)
        # fulldataset = fulldataset.append(doc_data, ignore_index=True)
        # training_set= pd.concat([combi_data, cust_combi_data], ignore_index=True)
        # fulldataset = fulldataset.append(combi_data, ignore_index=True)
        # fulldataset = fulldataset.append(cust_combi_data, ignore_index=True)
        del dataset
        gc.collect()
        # del doc_data
        # del combi_data
        # del cust_combi_data
        fulldataset.to_pickle(dest_pickle_file, compression='gzip')
        fulldataset.to_csv(dest_csv_file)
        with open(dest_dir + '/training_config.json', 'w') as outfile:
            json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)
        logging.warning('       Pickled ' + dest_pickle_file +'.')
        #del training_set
        


if __name__ == "__main__":
    main()





