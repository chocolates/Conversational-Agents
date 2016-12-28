import argparse
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os

from textdata import TextData
from wordfreq import WordFreq

class ChatBot:

    class TestMode:
        """Simple structure representing the different testing modes"""
        ALL = 'all'
        INTERACTIVE = 'interactive'
        DAEMON = 'daemon'

    def __init__(self):
        self.args = None

    @staticmethod
    def parseArgs(args):
        parser = argparse.ArgumentParser()
        #Global options
        globalArgs = parser.add_argument_group('Global options')
        globalArgs.add_argument('--test',nargs='?',
                                choices=[TestMode.ALL,TestMode.INTERACTIVE,TestMode.DAEMON], const=TestMode.ALL,default=None, help='if present, lauch the program to answer all sentences from a test file with or without target sentences; in interactive mode, user can write his or her own sentences; use daemon mode to integrate the chatbot into another program')
        globalArgs.add_argument('--rootDir', type=str, default=None, help='folder where to look for models and data')

        #Dataset options
        datasetArgs = parser.add_argument_group('Dataset options')
        datasetArgs.add_argument('--corpus', choices=['ubuntu','opensubs'], default='ubuntu', help='corpus on which to extract data')
        datasetArgs.add_argument('--trainFile', type=str, default='train.csv', help='data path for training text data')
        datasetArgs.add_argument('--validFile', type=str, default='valid.csv', help='data path for validation text data')
        datasetArgs.add_argument('--testFile', type=str, default='test.csv', help='data path for test text data')
        datasetArgs.add_argument('--dictSize', type=int, default=100000, help='maximum number of tokens to be considered as proper, all other less frequent tokens are considered unknown.')
        return parser.parse_args(args)

    def main(self, args=None):
        self.args = self.parseArgs(args)

        if not self.args.rootDir:
            self.args.rootDir = os.getcwd()

        self.wordfreq = WordFreq(self.args)
        self.trainData = TextData(self.args,'train')
        self.validData = TextData(self.args,'valid')
        if self.args.test==TestMode.ALL:
            self.testData = TextData(self.args,'test')

c = ChatBot().main()
