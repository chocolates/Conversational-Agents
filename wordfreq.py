'''
create sorted list of words based on their frequency in the entire corpus
STEPS:
	(1) count how many times each word appears => in self.word_frequence
	(2) sort (word, frequency) list => saved in self.sorted_words_list => save self.sorted_words_list in self.sorted_tuple_path = 'tuple.pkl'
Parameters:
	(1) corpus_dir := where is the corpus
	(2) list_path := where to save the sorted list
'''

import numpy as np
import nltk  # For tokenize
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random
import unicodecsv
import csv
import operator
import pickle

class WordFreq:

    def __init__(self, args):
        #model parameters
        self.args = args
        self.corpus = self.args.corpus
	if self.corpus=='ubuntu':
	    self.corpus_dir = os.path.join(self.args.rootDir,'data/ubuntu/dialogs/')
	else:
	    self.corpus_dir = os.path.join(self.args.rootDir, 'data/opensubs/') #to be modified
        self.dialog_path_list = []
        self.sorted_list_path = os.path.join(self.args.rootDir, self.corpus+'_freqlist.pkl')
        self.word_frequence = {}
        self.sorted_words_list = []

        self.load_sorted_list()

    def get_dialog(self, dialog_path):
        #print(dialog_path)
	dialog = []
	dialog_file = open(dialog_path, 'r')
	dialog_reader = unicodecsv.reader(dialog_file, delimiter='\t',quoting=csv.QUOTE_NONE)
	for dialog_line in dialog_reader:
	    context = dialog_line[3]
            # print context
	    dialog.append(context)
	return dialog

    def browse_all_subfolder_path(self):
	root_path = self.corpus_dir
	sub_folder_list = os.listdir(root_path)
	subfolder_path_list = []
	#print sub_folder_list
	for sub_folder_name in sub_folder_list:
	    if sub_folder_name == '.DS_Store':
		continue
	    sub_folder = sub_folder_name + '/'
	    subfolder_path_list.append(sub_folder)
	    # print sub_folder
	return subfolder_path_list

    def browse_all_file_path(self, subfolder_path_list):
        file_list = []
        for subfolder_path in subfolder_path_list:
	    file_path = self.corpus_dir + subfolder_path
	    all_files = os.listdir(file_path)
	    for file_name in all_files:
		if file_name == '.DS_Store':
		    continue
		file_list.append(subfolder_path + file_name)
        return file_list

    def construct_list(self):
        if self.corpus=='ubuntu':
            self.construct_list_ubuntu()
        else:
            self.construct_list_movies()

    def construct_list_movies(self):
        pass

    def construct_list_ubuntu(self):
	self.dialog_path_list = self.browse_all_file_path(self.browse_all_subfolder_path())
	for dialog_path in tqdm(self.dialog_path_list):
	    dialog_path = self.corpus_dir+dialog_path
	    dialog = self.get_dialog(dialog_path)
	    for sentence in dialog:
		word_list = nltk.word_tokenize(sentence)
		for word in word_list:
                    word = word.lower()
		    count = self.word_frequence.get(word, -1)
		    if count == -1:
			self.word_frequence[word] = 1
		    else:
			self.word_frequence[word] = count + 1
	self.sorted_words_list = sorted(self.word_frequence.items(), key=operator.itemgetter(1), reverse=True)

    def load_sorted_list(self):
        datasetExist = False
        if os.path.exists(self.sorted_list_path):
            datasetExist = True

        if not datasetExist:
            print('Frequency list not found. Creating sorted list...')
            self.construct_list()
            print('Saving sorted list...')
            self.save_sorted_list()
        else:
            print('Loading sorted list...')
            self.sorted_words_list = pickle.load(open(self.sorted_list_path, "rb" ))

    def save_sorted_list(self):
	''' [(word1, frequence1), (word2, frequence2)]'''
	data = self.sorted_words_list
	with open(self.sorted_list_path, 'wb') as handle:
	    pickle.dump(data, handle, -1)
