'''
create dictionary from given text corpus
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

class createDict:
	def __init__(self, args):
		self.args = args
		self.corpus_dir = '/Users/hanzhichao/Documents/ETH_Courses/DeepLearning/project/generate_ubuntu_data_set/dialogs/'
		self.dictionary_path = 'dictionary.txt'


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
			print context
			dialog.append(str(context))
		return dialog

	def construct_dict(self):
		word_frequence = {}



t = createDict('play')
subfolder_path = t.browse_all_subfolder_path()
print subfolder_path
# for i in subfolder_path:
# 	print i

file_path = t.browse_all_file_path(['236/'])
for i in file_path:
	print i

dialog = t.get_dialog(t.corpus_dir + '236/1.tsv')
print dialog