"""
Loads the dialogue corpus, builds the vocabulary
"""

import numpy as np
import nltk  # For tokenize
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random # for random number
import pandas as pd #data analysis tool

class TextData:
    def __init__(self, args):
        self.args = args
        self.data_path = 'data/train.csv'
        self.samples_path = 'data/samples.pkl'

        self.padToken = -1  # Padding
        self.goToken = -1  # Start of sequence
        self.eosToken = -1  # End of sequence
        self.unknownToken = -1  # Word dropped from vocabulary

        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]

        self.word2id = {}
        self.id2word = {}  # For a rapid conversion

        self.loadCorpus()

        # Plot some stats:
        print('Loaded: {} words, {} training samples'.format(len(self.word2id), len(self.trainingSamples)))

    def loadCorpus(self):
        """Load/create the conversations data
        Args:
            dirName (str): The directory where to load/save the model
        """
        datasetExist = False
        if os.path.exists(self.samples_path):
            datasetExist = True

        if not datasetExist:  # First time we load the database: creating all files
            print('Training samples not found. Creating dataset...')
            # Corpus creation
            self.createCorpus(self.data_path)

            # Saving
            print('Saving dataset...')
            self.saveDataset()  # Saving tf samples
        else:
            print('Loading dataset from ...')
            self.loadDataset()

        assert self.padToken == 0

    def createCorpus(self, data_path):
        """Extract all data from the given vocabulary
        """
        # Add standard tokens
        self.padToken = self.getWordId("<pad>")  # Padding (Warning: first things to add > id=0 !!)
        self.goToken = self.getWordId("<go>")  # Start of sequence
        self.eosToken = self.getWordId("<eos>")  # End of sequence
        self.unknownToken = self.getWordId("<unknown>")  # Word dropped from vocabulary

        # Remove __eou__ and __eot__ tags
        conversation = self.removeTag(data_path)

        # Iterate over rows in conversation dataframe

        for index in tqdm(range(1,len(conversation))):
            inputWords = self.extractText(nltk.word_tokenize(conversation.iloc[index]['Context'].decode('utf8')))
            targetWords = self.extractText(nltk.word_tokenize(conversation.iloc[index]['Utterance'].decode('utf8')))

            if inputWords and targetWords:
                self.trainingSamples.append([inputWords, targetWords])


    def extractText(self, words):
        wordIDs = []
        for word in words:
            wordIDs.append(self.getWordId(word))
        return wordIDs

    def getWordId(self, word, create=True):
        word = word.lower()
        wordID = self.word2id.get(word,-1)

        if wordID==-1:
            if create:
                wordID = len(self.word2id)
                self.word2id[word] = wordID
                self.id2word[wordID] = word
            else:
                wordID = self.unknownToken

        return wordID

    def removeTag(self, data_path):
        df = pd.read_csv('data/train.csv',header = 0, usecols = [0,1])
        df['Context'] = df['Context'].str.replace('__eou__','')
        df['Context'] = df['Context'].str.replace('__eot__','')
        df['Utterance'] = df['Utterance'].str.replace('__eou__','')
        df['Utterance'] = df['Utterance'].str.replace('__eot__','')
        return df

    def saveDataset(self):
        """Save samples to file
        Args:
            dirName (str): The directory where to load/save the model
        """

        with open(self.samples_path, 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                "word2id": self.word2id,
                "id2word": self.id2word,
                "trainingSamples": self.trainingSamples
                }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def loadDataset(self):
        """Load samples from file
        Args:
            dirName (str): The directory where to load the model
        """
        with open(self.samples_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word2id = data["word2id"]
            self.id2word = data["id2word"]
            self.trainingSamples = data["trainingSamples"]

            self.padToken = self.word2id["<pad>"]
            self.goToken = self.word2id["<go>"]
            self.eosToken = self.word2id["<eos>"]
            self.unknownToken = self.word2id["<unknown>"]  # Restore special words
'''
df = pd.read_csv('data/test.csv',header = 0, usecols = [0,1])
df['Context'] = df['Context'].str.replace('__eou__','')
df['Context'] = df['Context'].str.replace('__eot__','')
df['Utterance'] = df['Utterance'].str.replace('__eou__','')
df['Utterance'] = df['Utterance'].str.replace('__eot__','')
#print df.head()
print df.iloc[1]['Context']
print nltk.word_tokenize(df.iloc[1]['Context'])'''

t = TextData('play')
t.loadCorpus()
