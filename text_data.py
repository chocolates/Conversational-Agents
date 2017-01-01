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

MAX_CONTEXT_LENGTH = 100
MAX_UTTER_LENGTH = 50
_buckets = [(10,20),(20,40),(30,60),(40,80),(50,100)]

class TextData:

    def __init__(self, samples_path, data_path, sorted_list_path, dialog_path, vocab_size, playDataset = 0):
        self.padToken = '<pad>'
        self.goToken = '<go>'
        self.eosToken = '<eos>'
        self.unknownToken = '<unknown>'
        self.max_context_len = MAX_CONTEXT_LENGTH
        self.max_utter_len = MAX_UTTER_LENGTH
        self.word2id = {}
        self.id2word = {}
        self.samples = [[] for _ in _buckets]
        self.load_corpus(samples_path, data_path, sorted_list_path, dialog_path, vocab_size, playDataset)

    def load_corpus(self, samples_path, data_path, sorted_list_path, dialog_path, vocab_size, playDataset = 0):
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
            self.create_corpus(data_path, sorted_list_path, dialog_path, vocab_size)

            # Saving
            print('Saving dataset...')
            self.save_dataset(samples_path)  # Saving tf samples
        else:
            print('Loading dataset from ...')
            self.load_dataset(samples_path)

        assert self.word2id[self.padToken] == 0

        # Plot some stats:
        print('Loaded: {} words, {} training samples'.format(len(self.word2id), len(self.samples)))

        if playDataset>0:
            self.play_dataset(playDataset)

    def create_corpus(self, data_path, sorted_list_path, dialog_path, vocab_size):
        """Extract all data from the given vocabulary
        """
        self.initialize_vocabulary(sorted_list_path, dialog_path, vocab_size)

        # Remove __eou__ and __eot__ tags
        conversation = self.remove_tag(data_path)

        # Iterate over rows in conversation dataframe
        for index in tqdm(range(1,len(conversation))):
            inputWords = self.extract_text(nltk.word_tokenize(conversation.iloc[index]['Context'].decode('utf8','ignore')))
            targetWords = self.extract_text(nltk.word_tokenize(conversation.iloc[index]['Utterance'].decode('utf8','ignore')))

            if inputWords and targetWords:
                if len(inputWords)>self.max_context_len:
                    inputWords = inputWords[-self.max_context_len:]
                if len(targetWords)>self.max_utter_len:
                    targetWords = targetWords[:self.max_utter_len]
                for bucket_id, (input_size, target_size) in enumerate(_buckets):
                    if len(inputWords)<=input_size and len(targetWords)<=target_size:
                        targetWords.append(self.word2id[self.eosToken])
                        self.samples[bucket_id].append([inputWords, targetWords])
                        break

    def initialize_vocabulary(self, sorted_list_path, dialog_path, vocab_size):

        sorted_words_list = load_sorted_list(sorted_list_path, dialog_path)

        word_list = [x[0] for x in sorted_words_list[:vocab_size]]
        word_list.insert(0,self.unknownToken)
        word_list.insert(0,self.eosToken)
        word_list.insert(0,self.goToken)
        word_list.insert(0,self.padToken)
        id = range(0,len(word_list))
        self.word2id = dict(zip(word_list,id))
        self.id2word = dict(zip(id,word_list))


    def remove_tag(self, data_path):
        df = pd.read_csv(data_path,header = 0, usecols = [0,1])
        df['Context'] = df['Context'].str.replace('__eou__','')
        df['Context'] = df['Context'].str.replace('__eot__','')
        df['Utterance'] = df['Utterance'].str.replace('__eou__','')
        df['Utterance'] = df['Utterance'].str.replace('__eot__','')
        return df

    def extract_text(self, words):
        wordIDs = []
        for word in words:
            wordID = self.word2id.get(word.lower(),-1)
            if wordID==-1:
                wordIDs.append(self.word2id['<unknown>'])
            else:
                wordIDs.append(wordID)
        return wordIDs

    def save_dataset(self, samples_path):
        with open(samples_path, 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                "word2id": self.word2id,
                "id2word": self.id2word,
                "samples": self.samples
                }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def load_dataset(self, samples_path):
        with open(samples_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word2id = data["word2id"]
            self.id2word = data["id2word"]
            self.samples = data["samples"]


    def sequence2str(self, sequence, clean=False, reverse=False):
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
            return ' '.join([self.id2word[id] for id in sequence])

        sentence = []
        for id in sequence:
            if id==self.id2word[self.eosToken]:
                break
            elif id!=self.id2word[self.padToken] and id!=self.id2word[self.goToken]:
                sentence.append(self.id2word[id])

        return ' '.join(sentence)

    def play_dataset(self, num):
        """Print a random dialogue from the dataset
        """
        print('Randomly play samples:')
        for i in range(num):
            idBucket = random.randint(0,len(self.samples)-1)
            idSample = random.randint(0,len(self.samples[idBucket])-1)
            print('Context: {}'.format(self.sequence2str(self.samples[idBucket][idSample][0])))
            print('Utterance: {}'.format(self.sequence2str(self.samples[idBucket][idSample][1])))

t = TextData('ubuntu_train_samples.pkl', 'data/ubuntu/train.csv', 'ubuntu_freqlist.pkl', 'data/ubuntu/dialogs/', 50000, 20)
print len(t.samples)
