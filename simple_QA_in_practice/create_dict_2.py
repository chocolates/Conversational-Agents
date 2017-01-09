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

''' load word frequency list if already present,
	otherwise create one from raw ubuntu dialogue data
'''
def load_sorted_list(sorted_list_path=None, dialog_path=None):
	datasetExist = False

	if not sorted_list_path:
		sorted_list_path = os.getcwd()+'/ubuntu_freqlist.pkl'
	else:
		sorted_list_path += '/ubuntu_freqlist.pkl'

	if not dialog_path:
		dialog_path = os.getcwd()+'/data/ubuntu/dialogs/'

	if os.path.exists(sorted_list_path):
		datasetExist = True

	if not datasetExist:
	        print('Frequency list not found. Creating sorted list...')
		sorted_words_list = construct_list(dialog_path)
		print('Saving sorted list...')
		save_sorted_list(sorted_list_path, sorted_words_list)
	else:
		print('Loading sorted list...')
		sorted_words_list = pickle.load(open(sorted_list_path, "rb" ))

	return sorted_words_list

''' construct list of words sorted by frequency
	return a sorted list of tuples (word,frequency)
'''
def construct_list(dialog_path):
	dialog_path_list = browse_all_file_path(dialog_path, browse_all_subfolder_path(dialog_path))

	word_frequence = {}
	for d_path in tqdm(dialog_path_list):
		d_path = dialog_path+d_path
		dialog = get_dialog(d_path)
		for sentence in dialog:
			word_list = nltk.word_tokenize(sentence)
			for word in word_list:
				word = word.lower()
				count = word_frequence.get(word, -1)
				if count == -1:
					word_frequence[word] = 1
				else:
					word_frequence[word] = count + 1
	return sorted(word_frequence.items(), key=operator.itemgetter(1), reverse=True)

def browse_all_file_path(dialog_path, subfolder_path_list):
	file_list = []
	for subfolder_path in subfolder_path_list:
		file_path = dialog_path + subfolder_path
		all_files = os.listdir(file_path)
		for file_name in all_files:
			if file_name == '.DS_Store':
				continue
			file_list.append(subfolder_path + file_name)
	return file_list

def browse_all_subfolder_path(root_path):
	sub_folder_list = os.listdir(root_path)
	subfolder_path_list = []
	for sub_folder_name in sub_folder_list:
		if sub_folder_name == '.DS_Store':
			continue
		sub_folder = sub_folder_name + '/'
		subfolder_path_list.append(sub_folder)
	return subfolder_path_list

def get_dialog(dialog_path):
	dialog = []
	dialog_file = open(dialog_path, 'r')
	dialog_reader = unicodecsv.reader(dialog_file, delimiter='\t',quoting=csv.QUOTE_NONE)
	for dialog_line in dialog_reader:
	    context = dialog_line[3]
	    dialog.append(context)
	return dialog

def save_sorted_list(sorted_list_path, sorted_words_list):
	with open(sorted_list_path, 'wb') as handle:
		pickle.dump(sorted_words_list, handle, -1)
