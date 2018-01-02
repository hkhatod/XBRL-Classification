import logging
import os.path
import pickle
import pandas as pd
import itertools
from sqlalchemy import create_engine, Integer, String
from sqlalchemy.sql import text
from sqlalchemy.sql import bindparam

def get_categories(engine, s_nw, filename):
		query = text("""
				SELECT 	PQ.LOCAL_NAME, concat(PQ.LOCAL_NAME,',',string_agg(EQ.LOCAL_NAME, ',')) AS CHILDREN
				FROM	
						DTS_RELATIONSHIP S_DR
				JOIN	ELEMENT EP ON S_DR.FROM_ELEMENT_ID = EP.ELEMENT_ID
				JOIN	QNAME PQ ON PQ.QNAME_ID=EP.QNAME_ID
				JOIN	ELEMENT EE ON S_DR.TO_ELEMENT_ID = EE.ELEMENT_ID
				JOIN  	QNAME EQ ON EQ.QNAME_ID=EE.QNAME_ID

				WHERE
						S_DR.DTS_NETWORK_ID = :std_net
				GROUP BY
						PQ.LOCAL_NAME
					""", bindparams=[bindparam('std_net', value=s_nw, type_=Integer)])
		with engine.connect() as con:
			df = pd.read_sql_query(query, con)

		if filename == 'SFP':
			sfp_category ={}
			sfp_category['sfp'] = ('sfp','Assets','LiabilitiesAndStockholdersEquity')
			for index, row in df.iterrows():
				sfp_category[df['local_name'][index]] = tuple(df['children'][index].split(','))
			with open('sfp_categories.pickle','wb') as f:
				pickle.dump(sfp_category, f)
		elif filename =='SOI':
			soi_category = {}
			soi_category['soi'] = ('soi', 'EarningsPerShareBasic','EarningsPerShareBasicAndDiluted','EarningsPerShareDiluted','NetIncomeLossAvailableToCommonStockholdersDiluted','NetIncomeLossNetOfTaxPerOutstandingLimitedPartnershipUnitDiluted',
			'NetIncomeLossPerOutstandingGeneralPartnershipUnitNetOfTax','NetIncomeLossPerOutstandingLimitedPartnershipAndGeneralPartnershipUnitBasic','NetIncomeLossPerOutstandingLimitedPartnershipUnitBasicNetOfTax','WeightedAverageNumberOfDilutedSharesOutstanding')
			for index, row in df.iterrows():
				soi_category[df['local_name'][index]] = tuple(df['children'][index].split(','))
			with open('soi_categories.pickle','wb') as f:
				pickle.dump(soi_category, f)
		elif filename == 'SCF':
			scf_category = {}
			scf_category['scf'] = ('scf', 'CashAndCashEquivalentsPeriodIncreaseDecrease', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect')
			for index, row in df.iterrows():
				scf_category[df['local_name'][index]] = tuple(df['children'][index].split(','))
			with open('scf_categories.pickle','wb') as f:
				pickle.dump(scf_category,f)			


		logging.warning('Loaded Query with ' + str(df.shape[0]) + ' rows.')
		df.to_csv(filename + '_category.csv')
		df.to_pickle(filename+'_category.pickle', compression='gzip')
		logging.warning('Loaded Query to file rows.csv')


def main():
    e = create_engine('postgresql+psycopg2://khatodh:445$2gD%3@public.xbrl.us:5432/edgar_db')
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    statements = ('SFP', 'SOI', 'SCF')
    std_nt = {}
    std_nt['SFP'] = (30289066,) # SFP US GAAP 2017
    std_nt['SOI'] = (30289071,) # SOI US GAAP 2017
    std_nt['SCF'] = (30289063,) # CFS US GAAP 2017
    for i in range(0, 3):
        get_categories(e, std_nt[statements[i]], statements[i])

	


if __name__ == "__main__":
    main()