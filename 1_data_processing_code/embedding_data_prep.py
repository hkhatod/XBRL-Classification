import pandas as pd
import re
import datefinder
import logging



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
    s=str(s)
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
    s = re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ',s)

    return(s)


def main():
    filename = './entire_doc_distinct.csv'
    outfile = './clean_entire_doc_distinct.csv'
    chunksize = 5000
    filezise  = 10000000
    s=''
  
    with open(filename, "r") as r, open(outfile, "w") as w:
        for chunk in pd.read_csv(filename, sep ='|', chunksize=chunksize, quotechar='"'):
            s =  s + chunk['label'].apply(clean).str.cat(sep='. ')
            if len(s) > filezise:
                w.writelines(s)
                s=''
        w.writelines(s)


if __name__=="__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    main()