import os

imdb_dir = '/home/sourabh/experiments/NLP_TUto/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

# Read the review(text) into list and labels list
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()

            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
print("raw texts shape: ", len(texts))
print("raw labels shape: ", len(labels))
print("Sample text review: ", texts[0])

# Tokenizing the text of the raw imdb dataset

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100 # Cutt all review over 100 words
training_samples = 200 # As of now we are going to use only 200 review for training
validation_samples = 10000 # Low traning data and large validation data
max_words = 10000 # Consider only top 10000 words in the dataset

tokenizer = Tokenizer(num_words= max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
print("sequences shape: ", len(sequences))
print("Sample sequence: ", sequences[0])


word_index = tokenizer.word_index
print("lenght of word_index: ", len(word_index))
#print("word_index: ", word_index)

# Pad the sequences for common length
data = pad_sequences(sequences, maxlen= maxlen)

labels = np.asarray(labels)
print("Shape of data tnesor: ", data.shape)
print("Shape of label tensor: ", labels.shape)


# Suffle the data into random indeces
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


# Parse the GloVe word-embedding file
# Let’s parse the unzipped file (a .txt file) to build an index that maps words (as strings)
# to their vector representation (as number vectors).
glove_dir = "/home/sourabh/experiments/NLP_TUto/glove.6B"

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype= 'float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors ' % len(embeddings_index))

# Prepare GloVe embedding matrix
# Building the embedding matrix that we can load into Embedding layer.
# The shape of matrix must be (max_words, embedding_dim) i.e. (10000, 100)
# Note that index 0 isn’t supposed to stand for any
# word or token—it’s a placeholder

embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Build the model
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Embedding(max_words, embedding_dim, input_length= maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# Loading pre-trained word embeddings into the Embedding Layer created above
model.layers[0].set_weights([embedding_matrix]) # the pre-trained matrix to embedding layer
model.layers[0].trainable = False               # Because this is pre-trained model we do not want to train them

# Compile and Train the Model
model.compile(optimizer= 'rmsprop',
              loss= 'binary_crossentropy',
              metrics= ['acc'])

history = model.fit(x_train, y_train,
                    epochs = 10,
                    batch_size = 32,
                    validation_data= (x_val, y_val))

model.save_weights('pre_trained_glove_model.h5') # Save the model for testing

# Testing

# Tokenizing the test set

test_dir = os.path.join(imdb_dir, 'test')
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)