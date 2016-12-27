'''
create dictionary from given text corpus
STEPS:
	(1) count how many times each word appears => in self.word_frequence
	(2) sort (word, frequency) list => saved in self.sorted_words_list => save self.sorted_words_list in self.sorted_tuple_path = 'tuple.pkl'
	(3) construct dictionary from the sorted words list => in word2id => save self.word2id in self.dictionary_path = 'dictionary.pkl'
Parameters:
	(1) corpus_dir := where is the corpus
	(2) dictionary_path := where to save the dictionary
	(3) dict_size := the number of words in dictionary
MISC:
	(1) each word in dictionary is coded with 'unicodecsv', e.g. 'hi' -> u'hi', but could directly find by key 'hi' or by u'hi'
	(2) word id is begin at 0, in save_dict(self) function
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

class createDict:
	def __init__(self, args):
		self.args = args
		self.corpus_dir = '/Users/hanzhichao/Documents/ETH_Courses/DeepLearning/project/generate_ubuntu_data_set/dialogs/'
		self.dialog_path_list = []
		self.sorted_tuple_path = 'tuple.pkl'
		self.dictionary_path = 'dictionary.pkl'
		self.word_frequence = {}
		self.sorted_words_list = []
		self.word2id = {}
		self.dict_size = 100000


	def browse_all_subfolder_path(self):
		root_path = self.corpus_dir
		sub_folder_list = os.listdir(root_path)
		subfolder_path_list = []
		# print sub_folder_list
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

	def get_dialog(self, dialog_path):
		dialog = []
		dialog_file = open(dialog_path, 'r')
		dialog_reader = unicodecsv.reader(dialog_file, delimiter='\t',quoting=csv.QUOTE_NONE)
		for dialog_line in dialog_reader:
			context = dialog_line[3]
			# print context
			dialog.append(context)
		return dialog

	def construct_dict(self):
		self.word_frequence
		for dialog_path in tqdm(self.dialog_path_list):
			dialog_path = self.corpus_dir+dialog_path
			# print dialog_path
			dialog = self.get_dialog(dialog_path)
			for sentence in dialog:
				word_list = nltk.word_tokenize(sentence)
				# print word_list
				for word in word_list:
					count = self.word_frequence.get(word, -1)
					if count == -1:
						self.word_frequence[word] = 1
					else:
						self.word_frequence[word] = count + 1
		self.sorted_words_list = sorted(self.word_frequence.items(), key=operator.itemgetter(1), reverse=True)

	def save_tuple_list(self):
		''' [(word1, frequence1), (word2, frequence2)]'''
		data = self.sorted_words_list
		with open(self.sorted_tuple_path, 'wb') as handle:
			pickle.dump(data, handle, -1)

	def load_tuple_list(self):
		''' [(word1, frequence1), (word2, frequence2)]'''
		if os.path.isfile(self.sorted_tuple_path):
			self.sorted_words_list = pickle.load(open( self.sorted_tuple_path, "rb" ) )
		else:
			self.sorted_words_list = self.save_tuple_list()

	def save_dict(self):
		for i in tqdm(range(self.dict_size)):
			word_tuple = self.sorted_words_list[i]
			word = word_tuple[0]
			# word = str(word) # TODO: unicode
			self.word2id[word] = i
		with open(self.dictionary_path, 'wb') as handle:
			pickle.dump(self.word2id, handle, -1)

	def load_dict(self):
		self.word2id = pickle.load(open( self.dictionary_path, "rb" ) )







t = createDict('play')
subfolder_path = t.browse_all_subfolder_path()
# print subfolder_path
# for i in subfolder_path:
# 	print i

file_path = t.browse_all_file_path(subfolder_path)
# for i in file_path:
# 	print i

# dialog = t.get_dialog(t.corpus_dir + '236/1.tsv')
# print dialog
# print "mark!!!!"
# print dialog[0]

# t.dialog_path_list = ['/Users/hanzhichao/Documents/ETH_Courses/DeepLearning/project/generate_ubuntu_data_set/dialogs/236/1.tsv']
t.dialog_path_list = file_path
# t.construct_dict()
# t.save_tuple_list()
t.load_tuple_list()
# print word_list
t.save_dict()