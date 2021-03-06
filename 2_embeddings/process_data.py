from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import random
import os
import sys
sys.path.append('..')
import zipfile
import numpy as np
from six.moves import urllib
import tensorflow as tf
import utils
import json
import re

# Parameters for downloading data
DOWNLOAD_URL = 'http://mattmahoney.net/dc/'
EXPECTED_BYTES = 15785688      #31344016
DATA_FOLDER = './data/'
FILE_NAME = 'finstmt_corpus.zip'

def clean_str(s):
	#s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)  #re.sub(r"[^A-Za-z0-9:() !?\'\`]", "", s) # keep space, remove comma and strip other vs replave with space.

	s = re.sub(r"[^A-Za-z0-9$#@:(),!?\'\`]", " ", s)
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
	s = re.sub(r"\s+", ' ', s).strip()
	s = ' '.join(s.split())
	if s is '':
		s='-'
	return s.strip().lower()

def download(file_name, expected_bytes):
    """ Download the dataset text8 if it's not already downloaded """
    file_path = DATA_FOLDER + file_name
    if os.path.exists(file_path):
        print("Dataset ready")
        return file_path
    file_name, _ = urllib.request.urlretrieve(DOWNLOAD_URL + file_name, file_path)
    file_stat = os.stat(file_path)
    if file_stat.st_size == expected_bytes:
        print('Successfully downloaded the file', file_name)
    else:
        raise Exception('File ' + file_name +
                        ' might be corrupted. You should try downloading it with a browser.')
    return file_path

def read_data(file_path):
    """ Read data into a list of tokens 
    There should be 17,005,207 tokens
    """
    with zipfile.ZipFile(file_path) as f:
        words = clean_str(tf.compat.as_str(f.read(f.namelist()[0]))).split() 
        # tf.compat.as_str() converts the input into the string
    return words

def build_vocab(words):
    """ Build vocabulary of VOCAB_SIZE most frequent words """
    dictionary = dict()
    count = [('<PADS>', -1)]
    ''' get vocab size instead of using a fix size '''
    #count.extend(Counter(words).most_common(vocab_size - 1))
    count.extend(Counter(words).most_common())
    vocab_size=len(count)
    index = 0
    utils.make_dir('processed')
    with open('processed/vocab_1000.tsv', "w") as f:
        for word, _ in count:
            dictionary[word] = index
            # if index < 10000:
            #''' wrtie the whole vocab to dic '''
            f.write(word + "\n") 
            index += 1
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    with open(DATA_FOLDER  + '/vocabulary.json', 'w') as outfile: #was word_index
	    json.dump(dictionary, outfile, indent=4, ensure_ascii=False)
    return dictionary, index_dictionary, vocab_size


def convert_words_to_index(words, dictionary):
    """ Replace each word in the dataset with its index in the dictionary """
    return [dictionary[word] if word in dictionary else 0 for word in words]

def generate_sample(index_words, context_window_size):
    """ Form training pairs according to the skip-gram model. """
    for index, center in enumerate(index_words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        for target in index_words[max(0, index - context): index]:
            yield center, target
        # get a random target after the center wrod
        for target in index_words[index + 1: index + context + 1]:
            yield center, target

def get_batch(iterator, batch_size):
    """ Group a numerical stream into batches and yield them as Numpy arrays. """
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch

def process_data(batch_size, skip_window):
    file_path = download(FILE_NAME, EXPECTED_BYTES)
    words = read_data(file_path)
    dictionary, _, vocab_size = build_vocab(words)
    index_words = convert_words_to_index(words, dictionary)
    del words # to save memory
    single_gen = generate_sample(index_words, skip_window)
    return get_batch(single_gen, batch_size), vocab_size

def get_index_vocab():
    file_path = download(FILE_NAME, EXPECTED_BYTES)
    words = read_data(file_path)
    return build_vocab(words)

if __name__=="__main__":
    download(FILE_NAME,EXPECTED_BYTES)
