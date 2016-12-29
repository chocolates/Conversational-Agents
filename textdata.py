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

from wordfreq import WordFreq

class TextData:
    def __init__(self, args, mode):
        self.args = args
        self.corpus = self.args.corpus
        self.sorted_list_path = os.path.join(self.args.rootDir, self.corpus+'_freqlist.pkl')
        if self.corpus=='ubuntu':
            if mode=='train':
                self.data_path = os.path.join(self.args.rootDir,'data/ubuntu/train.csv')
                self.samples_path = os.path.join(self.args.rootDir,'ubuntu__train_samples.pkl')
            elif mode=='valid':
                self.data_path = os.path.join(self.args.rootDir,'data/ubuntu/valid.csv')
                self.samples_path = os.path.join(self.args.rootDir,'ubuntu_valid_samples.pkl')
            elif mode=='test':
                self.data_path = os.path.join(self.args.rootDir,'data/ubuntu/test.csv')
                self.samples_path = os.path.join(self.args.rootDir,'ubuntu_test_samples.pkl') 
        else:
            self.samples_path = os.path.join(self.args.rootDir,'data/opensubus_samples.pkl') #to be modified

        self.dictSize = self.args.dictSize

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

        if self.args.playDataset:
            self.playDataset()

    def loadCorpus(self):
        """Load/create the conversations data
        Args:
            dirName (str): The directory where to load/save the model
        """
        datasetExist = False
        if os.path.exists(self.samples_path):
            datasetExist = True

        if not datasetExist:  # First time we load the database: creating all files
            print('Samples not found. Creating dataset...')
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
        self.createDict()
        
        # Add standard tokens
        self.padToken = self.getWordId("<pad>")  # Padding (Warning: first things to add > id=0 !!)
        self.goToken = self.getWordId("<go>")  # Start of sequence
        self.eosToken = self.getWordId("<eos>")  # End of sequence
        self.unknownToken = self.getWordId("<unknown>")  # Word dropped from vocabulary

        # Remove __eou__ and __eot__ tags
        conversation = self.removeTag(data_path)

        # Iterate over rows in conversation dataframe

        for index in tqdm(range(1,len(conversation))):
            inputWords = self.extractText(nltk.word_tokenize(conversation.iloc[index]['Context'].decode('utf8','ignore')))
            targetWords = self.extractText(nltk.word_tokenize(conversation.iloc[index]['Utterance'].decode('utf8','ignore')))

            if inputWords and targetWords:
                self.trainingSamples.append([inputWords, targetWords])

    def createDict(self):
        wordFreq = WordFreq(self.args)
        wordFreqList = wordFreq.sorted_words_list
        wordList = [x[0] for x in wordFreqList[0:self.dictSize]]
        wordList.insert(0,'<unknown>')
        wordList.insert(0,'<eos>')
        wordList.insert(0,'<go>')
        wordList.insert(0,'<pad>')
        id = range(0,len(wordList))
        self.word2id = dict(zip(wordList,id))
        self.id2word = dict(zip(id,wordList))
        

    def extractText(self, words):
        wordIDs = []
        for word in words:
            wordIDs.append(self.getWordId(word))
        return wordIDs

    def getWordId(self, word, create=True):
        word = word.lower()
        wordID = self.word2id.get(word,-1)

        if wordID==-1:
            wordID = self.unknownToken

        return wordID

    def removeTag(self, data_path):
        df = pd.read_csv(self.data_path,header = 0, usecols = [0,1])
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
            if id==self.eosToken:
                break
            elif id!=self.padToken and id!=self.goToken:
                sentence.append(self.id2word[id])

        return ' '.join(sentence)

    def playDataset(self):
        """Print a random dialogue from the dataset
        """
        print('Randomly play samples:')
        for i in range(self.args.playDataset):
            idSample = random.randint(0,len(self.trainingSamples))
            print('Q: {}'.format(self.sequence2str(self.trainingSamples[idSample][0])))
            print('A: {}'.format(self.sequence2str(self.trainingSamples[idSample][1])))
            print()
