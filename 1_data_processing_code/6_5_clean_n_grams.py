#pylint: disable=I0011
#pylint: disable=C0111
#pylint: disable=C0301
#pylint: disable=C0304
#pylint: disable=C0103
#pylint: disable=W0312
#pylint: disable=W0105
#pylint: disable=C0330
#pylint: disable=E0611
#pylint: disable=E1129
#pylint: disable=E1101
#pylint: disable=W1202
#pylint: disable=
#pylint: disable=

import os
from os import listdir
from os.path import isfile, join
import logging
import pandas as pd

def main():
    path = './training/pickles/standard and documentation/'
    to_process_dir = path + 'custom_element_combination/'
    processed_dir = path + 'cleaned_custom_element_combination/'
    files = [f for f in listdir(to_process_dir) if isfile(join(to_process_dir, f))]
    for file in files:
        if file.endswith('.pickle'):
            if os.path.isfile(processed_dir + file):
                logging.warning('{} already processed.'.format(file))
            else:
                df = pd.read_pickle(to_process_dir + file, compression='gzip')
                df['length'] = df['element'].str.split().apply(len)
                if df['length'].count()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          > 0:
                    max_len = max(df['length'])
                    min_len = min(df['length'])
                    if max_len > 2:
                        df_clean = pd.concat([df.where(df['length'] == max_len-1).dropna().reindex(), df.where(df['length'] == max_len-2).dropna().reindex()], ignore_index=True)
                        df_clean.to_pickle(processed_dir + file, compression='gzip')
                        df_clean.to_csv(processed_dir + os.path.splitext(file)[0]+'.csv')

if __name__ == "__main__":
    main()
