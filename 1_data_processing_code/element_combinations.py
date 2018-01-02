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
import itertools


def split_data(element):
    words = element.split()
    k=0
    combi_data = pd.DataFrame(columns=['element'])
    for t in itertools.product(range(len('01')), repeat=len(words)-1):
        df1 = pd.DataFrame(columns=['element'])
        df1.loc[k] = [''.join([words[j]+t[j]*' ' for j in range(len(t))])+words[-1]]
        k+=1
        combi_data = combi_data.append(df1, ignore_index=True)
        del df1
    return combi_data

def get_std_elements(engine,nt):
    query = text("""
                SELECT  Q.LOCAL_NAME
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
    standard_elements['elements'] = standard_elements['local_name'].str.replace(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ')
    return standard_elements

def main():
    eng = create_engine('postgresql+psycopg2://khatodh:445$2gD%3@public.xbrl.us:5432/edgar_db')
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    networks = (30289066, 30289071, 30289063,)
    logging.warning('Loaded ' + str(len(networks))  + ' networks')
    se = get_std_elements(eng, networks)
    #get_data(se['local_name'][1])
    path = './training/pickles/standard and documentation/'
    for index, element in se.iterrows():
        file = path + 'element_combination/' + element['local_name'] 
        if os.path.isfile(file+ '.pickle'):
            logging.warning('Cleaning '+ element['local_name'])
            df = pd.read_pickle(file+ '.pickle', compression='gzip')
            #df = df.drop(df[df['element'].str.len() < df['category'].str.len()[0]].index).reset_index()
            df.to_pickle(file+ '.pickle', compression='gzip')
            df.to_csv(file+ '.csv')
            #logging.warning('Compressed ' +  element['local_name'] + ' already pickled.')
            #del df
        else:
            logging.warning('Processing ' + element['local_name'] +'.')
            df = split_data(element['elements'])
            df.to_pickle(file + '.pickle', compression='gzip')
            df.to_csv(file+ '.csv')
            logging.warning('Loaded ' + str(index+1) + ' of ' + str(se.shape[0]) + ' elements.')


if __name__ == "__main__":
    main()
# get_data('CashAndCashEquivalentsAtCarryingValue')
# temp = pd.read_pickle('CashAndCashEquivalentsAtCarryingValue.pickle')
# print(temp)
