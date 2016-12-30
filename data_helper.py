"""
Loads the dialogue corpus, builds the vocabulary
"""

import numpy as np
import nltk
from tqdm import tqdm
import pickle
import math
import os
import random
import pandas as pd

from create_dict import *

padToken = '<pad>'
goToken = '<go>'
eosToken = '<eos>'
unknownToken = '<unknown>'
MAX_CONTEXT_LENGTH = 100
MAX_UTTER_LENGTH = 50

def load_corpus(samples_path, data_path, sorted_list_path, dialog_path, vocab_size, playDataset = 0):
    """Load/create the conversations data
    Args:
        dirName (str): The directory where to load/save the model
    """
    datasetExist = False
    if os.path.exists(samples_path):
        datasetExist = True

    if not datasetExist:  # First time we load the database: creating all files
        print('Samples not found. Creating dataset...')
        # Corpus creation
        samples, word2id, id2word = create_corpus(data_path, sorted_list_path, dialog_path, vocab_size)

        # Saving
        print('Saving dataset...')
        save_dataset(samples_path, word2id, id2word, samples)  # Saving tf samples
    else:
        print('Loading dataset from ...')
        word2id, id2word, samples = load_dataset(samples_path)

    assert word2id[padToken] == 0

    # Plot some stats:
    print('Loaded: {} words, {} training samples'.format(len(word2id), len(samples)))

    if playDataset>0:
        play_dataset(playDataset, samples, id2word)

def create_corpus(data_path, sorted_list_path, dialog_path, vocab_size):
    """Extract all data from the given vocabulary
    """
    word2id, id2word = initialize_vocabulary(sorted_list_path, dialog_path, vocab_size)

    # Remove __eou__ and __eot__ tags
    conversation = remove_tag(data_path)

    # Iterate over rows in conversation dataframe
    samples = []
    for index in tqdm(range(1,len(conversation))):
        inputWords = extract_text(nltk.word_tokenize(conversation.iloc[index]['Context'].decode('utf8','ignore')), word2id)
        targetWords = extract_text(nltk.word_tokenize(conversation.iloc[index]['Utterance'].decode('utf8','ignore')), word2id)

        if inputWords and targetWords:
            if len(inputWords)>MAX_CONTEXT_LENGTH:
                inputWords = inputWords[-MAX_CONTEXT_LENGTH:]
            if len(targetWords)>MAX_UTTER_LENGTH:
                targetWords = targetWords[:MAX_UTTER_LENGTH]
            samples.append([inputWords, targetWords])

    return samples, word2id, id2word

def initialize_vocabulary(sorted_list_path, dialog_path, vocab_size):

    sorted_words_list = load_sorted_list(sorted_list_path, dialog_path)

    word_list = [x[0] for x in sorted_words_list[:vocab_size]]
    word_list.insert(0,unknownToken)
    word_list.insert(0,eosToken)
    word_list.insert(0,goToken)
    word_list.insert(0,padToken)
    id = range(0,len(word_list))
    word2id = dict(zip(word_list,id))
    id2word = dict(zip(id,word_list))

    return word2id, id2word

def remove_tag(data_path):
    df = pd.read_csv(data_path,header = 0, usecols = [0,1])
    df['Context'] = df['Context'].str.replace('__eou__','')
    df['Context'] = df['Context'].str.replace('__eot__','')
    df['Utterance'] = df['Utterance'].str.replace('__eou__','')
    df['Utterance'] = df['Utterance'].str.replace('__eot__','')
    return df

def extract_text(words, word2id):
    wordIDs = []
    for word in words:
        wordID = word2id.get(word.lower(),-1)
        if wordID==-1:
            wordIDs.append(word2id['<unknown>'])
        else:
            wordIDs.append(wordID)
    return wordIDs

def save_dataset(samples_path, word2id, id2word, samples):
    with open(samples_path, 'wb') as handle:
        data = {  # Warning: If adding something here, also modifying loadDataset
            "word2id": word2id,
            "id2word": id2word,
            "samples": samples
            }
        pickle.dump(data, handle, -1)  # Using the highest protocol available

def load_dataset(samples_path):
    with open(samples_path, 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        word2id = data["word2id"]
        id2word = data["id2word"]
        samples = data["samples"]
        return word2id, id2word, samples

def sequence2str(sequence, id2word, clean=False, reverse=False):
    """Convert a list of integer into a human readable string
    Args:
        sequence (list<int>): the sentence to print
        clean (Bool): if set, remove the <go>, <pad> and <eos> tokens
        reverse (Bool): for the input, option to restore the standard order
    Return:
        str: the sentence
    """
    if not sequence:
        return ''

    if reverse:
        sequence.reverse()

    if not clean:
        return ' '.join([id2word[id] for id in sequence])

    sentence = []
    for id in sequence:
        if id==id2word[eosToken]:
            break
        elif id!=id2word[padToken] and id!=id2word[goToken]:
            sentence.append(id2word[id])

    return ' '.join(sentence)

def play_dataset(num, samples, id2word):
    """Print a random dialogue from the dataset
    """
    print('Randomly play samples:')
    for i in range(num):
        idSample = random.randint(0,len(samples))
        print('Context: {}'.format(sequence2str(samples[idSample][0], id2word)))
        print('Utterance: {}'.format(sequence2str(samples[idSample][1], id2word)))

load_corpus('ubuntu_valid_samples', 'data/ubuntu/valid.csv', 'ubuntu_freqlist.pkl', 'data/ubuntu/dialogs', 50000, 20)
