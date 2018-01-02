# pylint: disable=I0011
# pylint: disable=C0111
# pylint: disable=C0301
# pylint: disable=C0103
#import re
''' 
getdata gets all parent child relations from all the standard taxonamoies and mapped to 2017(most current taxonomies). 
This allows to to capture least common denomidator of all possible relationship in all the standard taxonamy with respect 
to 2017 taxonamy.

create_training_data flattens out the hierarchy.


'''

import logging
import os.path
import pickle
import pandas as pd
from sqlalchemy import create_engine, Integer, String
from sqlalchemy.sql import text
from sqlalchemy.sql import bindparam
from input_dataset import networks, sfp_category, soi_category, scf_category
def getdata(engine, nw, s_nw, root_p): # Gets raw data from Public database- Parent child relationships within networks.
    query = text("""WITH RECURSIVE CTE (ROW_NUM, DEPTH, NETWORK_ID, RELATIONSHIP_ID, R_DEPTH, R_SEQ, R_ORDER, CHILD_ID, CHILD_NAME, PARENT_ID, PARENT_NAME, PATH) as
                        (	
                            (
                            SELECT 
                                row_number() over()
                                ,1
                                ,S_DR.DTS_NETWORK_ID
                                ,DR.DTS_RELATIONSHIP_ID AS RELATIONSHIP
                                ,S_DR.TREE_DEPTH
                                ,S_DR.TREE_SEQUENCE
                                ,S_DR.RELN_ORDER
                                ,EE.ELEMENT_ID as CHILD_ID
                                ,EQ.LOCAL_NAME as CHILD_NAME  
                                ,S_DR.FROM_ELEMENT_ID as PARENT_ID
                                ,PQ.LOCAL_NAME as PARENT_NAME
                                ,ARRAY[PQ.LOCAL_NAME]
                            FROM
                                DTS_NETWORK DN
                                ,DTS_RELATIONSHIP DR
                                ,ELEMENT EE
                                ,ELEMENT EP
                                ,QNAME EQ
                                ,QNAME PQ
                          		,DTS_RELATIONSHIP S_DR
		                        ,QNAME S_EQ
                        		,ELEMENT S_EE
                            WHERE	
                                DN.DTS_NETWORK_ID=DR.DTS_NETWORK_ID
                                AND EE.ELEMENT_ID=DR.TO_ELEMENT_ID
                                AND EP.ELEMENT_ID=S_DR.FROM_ELEMENT_ID
                                AND EQ.QNAME_ID=EE.QNAME_ID
                                AND PQ.QNAME_ID=EP.QNAME_ID
                                AND S_EE.ELEMENT_ID=S_DR.TO_ELEMENT_ID
                                AND S_EE.QNAME_ID=S_EQ.QNAME_ID
                                AND S_EQ.LOCAL_NAME=EQ.LOCAL_NAME
                                AND DN.DTS_NETWORK_ID IN :network
                                AND S_DR.DTS_NETWORK_ID = :std_net
                                --AND PQ.LOCAL_NAME = 'Assets'
                            ORDER BY
                                DR.TREE_SEQUENCE
                            )
                            UNION 
                            (
                            SELECT 	
                                CTE.ROW_NUM
                                ,CTE.DEPTH + 1
                                ,R_DR.DTS_NETWORK_ID
                                ,R_DR.DTS_RELATIONSHIP_ID
                                ,R_DR.TREE_DEPTH
                                ,R_DR.TREE_SEQUENCE
                                ,R_DR.RELN_ORDER
                                ,R_EE.ELEMENT_ID as CHILD_ID
                                ,R_EQ.LOCAL_NAME as CHILD_NAME
                                ,R_DR.FROM_ELEMENT_ID as R_PARENT_ID
                                ,R_PQ.LOCAL_NAME as PARENT_NAME
                                ,PATH || R_PQ.LOCAL_NAME
                            FROM		
                                DTS_RELATIONSHIP R_DR 
                                JOIN CTE ON R_DR.DTS_NETWORK_ID=CTE.NETWORK_ID
                                JOIN ELEMENT R_EE ON R_EE.ELEMENT_ID=CTE.PARENT_ID
                                JOIN ELEMENT R_EP ON R_DR.FROM_ELEMENT_ID=R_EP.ELEMENT_ID
                                JOIN QNAME R_EQ ON R_EE.QNAME_ID=R_EQ.QNAME_ID
                                JOIN QNAME R_PQ ON R_EP.QNAME_ID=R_PQ.QNAME_ID
                                
                            WHERE	
                                R_DR.TO_ELEMENT_ID=R_EE.ELEMENT_ID
                            ORDER BY
                                R_DR.TREE_SEQUENCE
                            )
                        )
                        SELECT 
				                --C.ROW_NUM as ROW_NUM, P.DEPTH as depth, 
                                C.CHILD_NAME as CHILD_NAME, P.PATH as PATH
                        FROM	
                                CTE C
                                ,CTE P
			            WHERE
                                C.ROW_NUM=P.ROW_NUM
                                AND C.R_DEPTH=P.DEPTH
                                AND C.DEPTH=1
                               --AND P.PARENT_NAME=:root
			            --ORDER BY
				          --      ROW_NUM, DEPTH
                """, bindparams=[bindparam('network', value=nw, type_=Integer), bindparam('std_net', value=s_nw, type_=Integer), bindparam('root', value=root_p, type_=String)])
    with engine.connect() as con:
        rows = pd.read_sql_query(query, con)
    logging.warning('Loaded Query with ' + str(rows.shape[0]) + ' rows.')
    rows.to_csv('rows.csv')
    logging.warning('Loaded Query to file rows.csv')
    
    #rows = pd.read_csv('rows.csv')
    return rows

## end of getdata()


# def create_training_data(pc, categories): # creates training sets
#     k = 1
#     training_set = {}
#     df = pd.DataFrame(columns=['category', 'element'])

#     for index, row in pc.iterrows():
#         unclassified = True
#         for category in categories[1:]:
#             # if category != categories[0]:
#             if category in row['path']:
#                 #df.set_value(k, 'relationship', row['relationship'])
#                 df.set_value(k, 'category', category)
#                 df.set_value(k, 'element', re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', row['child_name']))
#                 unclassified = False
#                 k += 1
#         if unclassified is True and categories[0] in row['path']:
#             df.set_value(k, 'category', categories[0])
#             df.set_value(k, 'element', re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', row['child_name']))
#             k += 1
#         #logging.warning('Completed rows' +str(index) +' of ' +str(pc.shape[0]))
#     training_set[categories[0]] = df
#     training_set[categories[0]].to_csv('trainingg_set.csv')
#     logging.warning("Wrote training set to training_set.csv with " +str(training_set[categories[0]].shape[0]) + ' rows' )
# # end create_training_data

def create_training_data(pc, categories):
    training_data = pd.DataFrame(columns=['category', 'element'])
    for category in reversed(categories):
        df = pd.DataFrame(columns=['category', 'element'])
        df['element'] = pc[pc['path'].str.contains('\''+category+'\'') & ~pc['classified']]['child_name'].reset_index(drop=True)
        df['category'] = category
        pc.loc[(pc['path'].str.contains('\''+category+'\'')) & (~pc['classified']), 'classified'] = True
        training_data = training_data.append(df, ignore_index=True)
        del df
    #training_data['element'] = training_data['element'].str.replace(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ')
    #logging.warning('Created training set for '+ categories[0] + ' with ' +str(training_data.shape[0]) + ' rows')
    #training_data.to_csv('training.csv')
    return training_data

def define_statement(eng, s_n, n, stmt, class_categories):
    path = './training/pickles/standard and documentation/'
    stmt_file = path +stmt + '.pickle'
    training_set_path = path + stmt
    if os.path.isfile(stmt_file):
        logging.warning('Loading pickle ' + stmt_file)
        par_cld = pd.read_pickle(stmt_file, compression='gzip')
        #par_cld['child_name'] = par_cld['child_name'].str.replace(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ')
        par_cld.to_csv(path +stmt + '.csv',sep='|')
        logging.warning('Loaded pickle ' + stmt + '.pickle with ' + str(par_cld.shape[0]) + ' rows.')
    else:
        logging.warning('Pickle '+ stmt_file +' does not exit. Quering the database.')
        logging.warning('Loaded ' + str(len(n))  + ' network names')
        par_cld = getdata(eng, n, s_n, '') # cats[0])
        par_cld['path'] = '\'' + par_cld['path'].apply(lambda x: '\',\''.join(x)) + '\''
        par_cld.to_pickle(stmt_file, compression='gzip')
        logging.warning('Pickled ' + stmt_file)

    for key, cats in class_categories.items():
        filename = training_set_path +'/'+ cats[0] +'.pickle'
        csv_filename = training_set_path +'/'+ cats[0] +'.csv'
        if os.path.isfile(filename):
            dataset = pd.read_pickle(filename, compression='gzip')
            dataset.to_csv(csv_filename)
            logging.warning('Loading ' + filename)
        else:
            par_cld['classified'] = False
            dataset = create_training_data(par_cld, cats)
            dataset.to_pickle(filename, compression='gzip')
            dataset.to_csv(csv_filename)
        
        del dataset
    del par_cld

def main():
    e = create_engine('postgresql+psycopg2://khatodh:445$2gD%3@public.xbrl.us:5432/edgar_db')
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    statements = ('SFP', 'SOI', 'SCF')
    std_nt = {}
    std_nt['SFP'] = (30289066,) # SFP US GAAP 2017
    std_nt['SOI'] = (30289071,) # SOI US GAAP 2017
    std_nt['SCF'] = (30289063,) # CFS US GAAP 2017
    nt = {}
    nt['SFP'] = networks['standard_sfp']# + networks['SANDP500_sfp']#[1:5000]#(29594697,)
    nt['SOI'] = networks['standard_soi']# + networks['SANDP500_soi']#[1:5000]#(29594697,)
    nt['SCF'] = networks['standard_scf']# + networks['SANDP500_scf']#[1:5000]#(29594697,)
    cc = {}
    cc['SFP'] = sfp_category
    cc['SOI'] = soi_category
    cc['SCF'] = scf_category
    for i in range(0, 3):
        define_statement(e, std_nt[statements[i]], nt[statements[i]], statements[i], cc[statements[i]])

if __name__ == "__main__":
    main()