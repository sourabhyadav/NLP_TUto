import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

print("Token Inxex: ", token_index)
max_length = 10

results = np.zeros(shape=(len(samples),
                          max_length,
                          max(token_index.values()) + 1))
print("results shape: ", results.shape)

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        print("word: ", word, " j: ", j, " index: ", index)
        results[i, j, index] = 1.

print("Vectorizd Results: ", results)
print("Result vector shape: ", results.shape)

# Keras word-level one-hot encoding
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words= 1000)
tokenizer.fit_on_texts(samples)
print("tokenizer: ", tokenizer)

seq = tokenizer.texts_to_sequences(samples)
print("seq: ", seq)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
print("one_hot res : ", one_hot_results, " shape: ", one_hot_results.shape)

word_index =tokenizer.word_index
print("Word_indx: ", word_index, " Found unique tokens: ", len(word_index))