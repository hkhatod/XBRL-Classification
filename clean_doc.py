import re
import pickle
import datefinder
import os
from os import listdir
from os.path import isfile, join
import logging
import pandas as pd

def clean(s):
    """
    Clean the documentation files. 
     - Replaces all dollar amounts with '$$$'
     - Replaces all dates with '@@@'
     - Replaces all remaining numbers 4,3,2,1 digits with '###'

     Params:
     ------
     s - string. lambda fucntion can be applied to entire 
     column of dataframe with dtype of string.

     Returns:
     -------
     s - cleaned string where all dollar amounts, dates, and numbers 
     are conversted to $$$, @@@, and ### respectively.
    """
    s = re.sub(r"[^A-Za-z0-9:(),!?$\'\`]", " ", s)
    s = re.sub(r"[\$]{1}[\d,]+\.?\d{0,2}"," $$$ ",s,0)  # Replace all dollar amounts with "$$$"
    matches = datefinder.find_dates(s,source=True,index=True)
    for match in matches:
        s = s.replace(match[1]," @@@ ") # Replace all dates with "@@@"
        s = re.sub(r"19|20\d{2}"," @@@ ",s,0) # Replace all years without full date with "@@@""
    s = re.sub(r"\d{6}|\d{5}|\d{4}|\d{3}|\d{2}|\d{1}"," ### ",s) # Replace all remaining numbers with  "###"
    s = re.sub(r"[^A-Za-z0-9:(),!?$#@\'\`]", " ", s)  #re.sub(r"[^A-Za-z0-9:() !?\'\`]", "", s) # keep space, remove comma and strip other vs replave with space.
    s = re.sub(r" : ", ":", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)

    return(s)

def tidy_split(df, column, sep='|', keep=False):
    """
    Split the values of a column and expand so the new DataFrame has one split
    value per row. Filters rows where the column is missing.

    Params
    ------
    df : pandas.DataFrame
        dataframe with the column to split and expand
    column : str
        the column to split and expand
    sep : str
        the string used to split the column's values
    keep : bool
        whether to retain the presplit value as it's own row

    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the same columns as `df`.

    Finally truncate at length 30 
    """
    indexes = list()
    new_values = list()
    df = df.dropna(subset=[column])
    for i, presplit in enumerate(df[column].astype(str)):
        values = presplit.split(sep)
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value)
    new_df = df.iloc[indexes, :].copy()
    new_df[column] = new_values
    return new_df


def main():
    path = './training/pickles/standard and documentation/'
    cleaned_path = path  + 'cleaned_docs/'
    files = [f for f in listdir(path+'documentation/') if isfile(join(path+'documentation/', f))]
    for file in files:
        if file.endswith('.pickle'):
            if os.path.isfile(cleaned_path+file):
                logging.warning(file +' already cleaned.')
            else:
                doc = pd.read_pickle(path+'documentation/'+file,compression='gzip')
                doc['element'] = doc['element'].apply(clean)
                #doc = tidy_split(doc,'element',sep='.')
                """
                Following line removes all rows of from the 
                document that has more than 200 charaters ~40 words.
                """
                #doc = doc[doc['element'].str.len() < 200].reset_index()  #take full sentences for 
                doc.to_csv(cleaned_path + os.path.splitext(file)[0]+'.csv') 
                doc.to_pickle(cleaned_path + os.path.splitext(file)[0]+'.pickle', compression='gzip') 
                logging.warning('Cleaned ' + file + '.')
                del doc

if __name__=="__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    main()
