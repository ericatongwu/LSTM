"""
LSTM Prediction Model for debate statement 
We will create N-grams sequence as predictors and the next word of the N-gram as label.
"""
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import numpy as np
import json 
from sklearn.model_selection import train_test_split

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
		input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')) # use a matrix to represent the word
		# pad_sequences is adding padding to make it into the same length
		predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
		
		label = ku.to_categorical(label, num_classes=total_words)     # ku.to_categorical is converting a class vector (integers) to binary class matrix.
		X_train, X_test, y_train, y_test = train_test_split(predictors, label, test_size=0.33, random_state=42) # split traing and testing data
	#return predictors, label, total_words, max_sequence_len
	return X_train, X_test, y_train, y_test, total_words, max_sequence_len

"""

1. Input Layer : Takes the sequence of words as input
2. LSTM Layer : Computes the output using LSTM units. I have added 100 units in the layer, but this number can be fine tuned later.
3. Dropout Layer : A regularisation layer which randomly turns-off the activations of some neurons in the LSTM layer. It helps in preventing over fitting.
4. Output Layer : Computes the probability of the best possible next word as output

"""
# Input: predictors, label, max_sequence_len, total_words
# Output: model

def create_model(X_train, X_test, y_train, y_test, max_sequence_len, total_words):
	input_len = max_sequence_len - 1

	model = Sequential()
	model.add(Embedding(total_words, 10, input_length=input_len))
	# The model will take as input an integer matrix of size (batch,input_length). batch = 10
	# The largest integer (i.e. word index) in the input should be no larger than total_words (vocabulary size).
	# Now model.output_shape == (None, 10, 64), where None is the batch
	model.add(LSTM(150))
	model.add(Dropout(0.1))
	model.add(Dense(total_words, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam')
	model.fit(X_train, y_train, epochs=100, verbose=1)
    # To modify, we could change epoches, loss fuction
	score = model.evaluate(X_test, y_test)
	print(score)
	return model

# Input: Input txt, wo
# Output:

def generate_text(seed_text, next_words, max_sequence_len, model):    # seed_text = inout words
	for j in range(next_words):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')    

		predicted = model.predict_classes(token_list, verbose=0)
		output_word = ""
		for word, index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		seed_text += " " + output_word

	return seed_text

X, Y, max_len, total_words = dataset_preparation(read_file('final.json'))
model = create_model(X, Y, max_len, total_words)
test1 = "I want thank you all for coming out. This is a very important question at this very difficult time in our nation's"
text = generate_text(test1, 3, msl, model)

print(text)