from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import numpy as np
import json 

# Input: json file
# Output: list of dictionary 

def read_file(file_name):
	with open(file_name) as f:
		data = json.load(f)
	return data['corpus']

# Input: string
# Output: tokenized string

tokenizer = Tokenizer()

def dataset_preparation(data):
	for dic in data:
		string = dic['data']
		corpus = string.lower().split("\n")
		tokenizer.fit_on_texts(corpus)
		total_words = len(tokenizer.word_index) + 1
		input_sequences = []
		for line in corpus:
			token_list = tokenizer.texts_to_sequences([line])[0]  # word to number which is index in corpus
			for i in range(1, len(token_list)):
				n_gram_sequence = token_list[:i+1]   # n+1 gram????
				input_sequences.append(n_gram_sequence)
				# make equal sequences important
				max_sequence_len = max([len(x) for x in input_sequences])
				input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
				# pad_sequences is adding padding to make it into the same length
# Input: 
# Output:

def create_model():
    pass

# Input:
# Output:

def generate_text():
    pass

dataset_preparation(read_file('final.json'))