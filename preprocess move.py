# pylint: disable=I0011
# pylint: disable=C0111
# pylint: disable=C0301
# pylint: disable=C0103
import logging
import os
from os import listdir
from os.path import isfile, join
import pickle
import shutil

def main():
    from_dir = './training/pickles/standard and documentation/custom_elements/'
    to_dir = './training/pickles/standard and documentation/custom_elements/to_move/'
    check_dir = './training/pickles/standard and documentation/custom_element_combination/'

    files = [f for f in listdir(from_dir) if isfile(join(from_dir, f))]
    for file in files:
        if file.endswith('.pickle'):
            if os.path.isfile(check_dir+file):
                logging.warning(file +' already processed.')
            else:
                shutil.move(from_dir+file, to_dir+file)
                shutil.move(from_dir+os.path.splitext(file)[0]+'.csv',to_dir+os.path.splitext(file)[0]+'.csv') 
    
if __name__=="__main__":
    main()