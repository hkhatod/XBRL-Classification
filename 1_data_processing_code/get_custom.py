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
from  input_dataset import sfp_category


def get_data(engine,ln):
    logging.warning('quering {}'.format(ln)) 
    query = text("""
                SELECT DISTINCT QP.LOCAL_NAME as category, Q.LOCAL_NAME as element, L.LABEL as documentation
                FROM 
                    DTS D
                    ,DTS_NETWORK DN
                    ,DTS_RELATIONSHIP R
                    ,DTS_ELEMENT DE
                    ,ELEMENT E
                    ,QNAME Q
                    ,ELEMENT EP
                    ,QNAME QP
                    ,DTS_NETWORK DNL
                    ,DTS_RELATIONSHIP DRL
                    ,LABEL_RESOURCE L
                WHERE
                    D.DTS_ID=DN.DTS_ID
                    AND DE.IS_BASE= FALSE
                    AND DN.DTS_NETWORK_ID = R.DTS_NETWORK_ID
                    AND DE.ELEMENT_ID = E.ELEMENT_ID
                    AND DE.DTS_ID=D.DTS_ID
                    AND R.TO_ELEMENT_ID = E.ELEMENT_ID
                    AND R.FROM_ELEMENT_ID =EP.ELEMENT_ID
                    AND EP.QNAME_ID=QP.QNAME_ID
                    AND E.QNAME_ID=Q.QNAME_ID
                    AND DNL.DTS_ID=D.DTS_ID
                    AND DNL.EXTENDED_LINK_QNAME_ID = 18998
                    AND DNL.DTS_NETWORK_ID=DRL.DTS_NETWORK_ID
                    AND DRL.FROM_ELEMENT_ID=R.TO_ELEMENT_ID
                    AND DRL.TO_RESOURCE_ID=L.RESOURCE_ID
                    AND QP.LOCAL_NAME =:parent
                ORDER BY
                    ELEMENT
                """, bindparams=[bindparam('parent', value=ln, type_=String)])
    
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
    return standard_elements

def main():
    eng = create_engine('postgresql+psycopg2://khatodh:445$2gD%3@public.xbrl.us:5432/edgar_db')
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    networks = (30289066, 30289071, 30289063)
    
    logging.warning('Loaded ' + str(len(networks))  + ' networks')
    #se = get_std_elements(eng, networks)
    se = pd.DataFrame.from_records(zip(list(sfp_category.keys())))
    se.columns =['local_name']
    #get_data(se['local_name'][1])

    for index, element in se.iterrows():
        path='./training/pickles/standard and documentation/custom_elements_processed/'
        if os.path.isfile(path + element['local_name'] + '.pickle'):
            # logging.warning('Already loaded '+ element['local_name'])
            # df = pd.read_pickle(path + 'custom_elements/' + element['local_name'] + '.pickle', compression='gzip')
            # df = df.drop_duplicates(subset=['category', 'element'])
            # df.to_pickle(path + 'custom_elements/' + element['local_name'] + '.pickle', compression='gzip')
            # df.to_csv(path + 'custom_elements/' + element['local_name'] + '.csv')
            logging.warning('We have '+ element['local_name'])
        else:
            #logging.warning('Processing ' + element['local_name'] +'.')
            
            df = get_data(eng, element['local_name'])
            df = df.drop_duplicates(subset=['category', 'element']).reset_index()
            #df = df.groupby(['network','local_name'], sort=False)['documentation'].apply('. '.join).reset_index()
            df.to_pickle(path +element['local_name'] + '.pickle', compression='gzip')
            df.to_csv(path + element['local_name'] + '.csv')
            logging.warning('Loaded ' + str(index+1) + ' of ' + str(se.shape[0]) + ' elements.')

if __name__ == "__main__":
    main()
# get_data('CashAndCashEquivalentsAtCarryingValue')
# temp = pd.read_pickle('CashAndCashEquivalentsAtCarryingValue.pickle')
# print(temp)
