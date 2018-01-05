# pylint: disable=I0011
# pylint: disable=C0111
# pylint: disable=C0301
# pylint: disable=C0103
import logging
import os.path
import pickle
import pandas as pd
from sqlalchemy import create_engine, Integer, String
from sqlalchemy.sql import text
from sqlalchemy.sql import bindparam


def get_data(engine,ln):
    query = text("""
                SELECT 
                        R.DTS_NETWORK_ID as network, Q.LOCAL_NAME as local_name, LR.LABEL as documentation
                FROM 
                        DTS_RELATIONSHIP R
                        ,ELEMENT E
                        ,QNAME Q
                        ,LABEL_RESOURCE LR
                WHERE
                        R.FROM_ELEMENT_ID = E.ELEMENT_ID
                        AND E.QNAME_ID = Q.QNAME_ID
                        AND R.TO_RESOURCE_ID = LR.RESOURCE_ID
                        AND Q.LOCAL_NAME =:local_name
                            
                """, bindparams=[bindparam('local_name', value=ln, type_=String)])
    with engine.connect() as con:
        labels = pd.read_sql_query(query, con)
    #logging.warning('Loaded labels Query with ' + str(labels.shape[0]) + ' rows.')
    #labels['category'] = ln
    #labels = labels[['category', 'element']]
    #labels.to_pickle('./training/pickles/documentation/'+ln + '.pickle',  compression='gzip')
    logging.warning('Pickled ' + ln + ' with '+ str(labels.shape[0]) + ' rows.')
    return labels
    #print(labels)


def get_std_elements(engine,nt):
   
    query = text("""
                SELECT  DISTINCT Q.LOCAL_NAME
                FROM
                        DTS_RELATIONSHIP R
                        ,ELEMENT E
                        ,QNAME Q
                WHERE
                        R.TO_ELEMENT_ID = E.ELEMENT_ID
                        AND E.QNAME_ID = Q.QNAME_ID
                        AND R.DTS_NETWORK_ID in :network
                """, bindparams=[bindparam('network', value=nt, type_=Integer)])
    with engine.connect() as con:
        standard_elements = pd.read_sql_query(query, con)
    logging.warning('Loaded elements query with ' + str(standard_elements.shape[0]) + ' rows.')
    return standard_elements

def main():
    eng = create_engine('postgresql+psycopg2://khatodh:445$2gD%3@public.xbrl.us:5432/edgar_db')
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    networks = (30276006,30277865,30279678,30281533,30283360,30285178,30287073,30289066,
            	30276017,30277882,30279695,30281550,30283365,30285183,30287078,30289071,
                30275991,30277848,30279661,30281516,30283357,30285175,30287070,30289063,)
    logging.warning('Loaded ' + str(len(networks))  + ' networks')
    se = get_std_elements(eng, networks)
    #get_data(se['local_name'][1])
    path = './training/pickles/standard and documentation/'
    for index, element in se.iterrows():
        if os.path.isfile(path + 'documentation/' + element['local_name'] + '.pickle'):
            logging.warning('Cleaning '+ element['local_name'])
            # df = pd.read_pickle(path + 'documentation/' + element['local_name'] + '.pickle', compression='gzip')
            # #df = df.drop(df[df['element'].str.len() < df['category'].str.len()[0]].index).reset_index()
            # df.to_pickle(path + 'documentation/' + element['local_name'] + '.pickle', compression='gzip')
            # df.to_csv(path + 'documentation/' + element['local_name'] + '.csv')
            #logging.warning('Compressed ' +  element['local_name'] + ' already pickled.')
            #del df
        else:
            logging.warning('Processing ' + element['local_name'] +'.')
            df = get_data(eng, element['local_name'])
            #df = df.drop_duplicates(subset=['network','local_name','documentation'])
            df = df.drop_duplicates(subset=['local_name','documentation'])
            df = df.groupby(['network','local_name'], sort=False)['documentation'].apply('. '.join).reset_index()
            #df = df.groupby(['network','local_name'])['documentation'].apply(lambda x: ' '.join(x)).reset_index()
            df = df.drop('network', 1)
            df = df.rename(columns={'local_name': 'category', 'documentation': 'element'})
            #df = df.drop(df[df['element'].str.len() < df['category'].str.len()[0]].index).reset_index()
            df.to_pickle(path + 'documentation/' + element['local_name'] + '.pickle', compression='gzip')
            df.to_csv(path + 'documentation/' + element['local_name'] + '.csv')
            logging.warning('Loaded ' + str(index+1) + ' of ' + str(se.shape[0]) + ' elements.')


if __name__ == "__main__":
    main()
# get_data('CashAndCashEquivalentsAtCarryingValue')
# temp = pd.read_pickle('CashAndCashEquivalentsAtCarryingValue.pickle')
# print(temp)
