

from keras.datasets import imdb
from keras import preprocessing

max_features = 10000    # Maximum feature to
maxlen = 20             # Will limit the review to max these words

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features) # This loads integer data
print("x_train shape: ", x_train.shape)
print("sample train data:", x_train[0])
print("Sample train label", y_train[0]) # 0 -> Negative Review, 1 -> Positive Review

# Restricting maximum length to maxlen for each review
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
print("After limiting x_train shape: ", x_train.shape)

# Build the Embedding Layer and Classifier Layer
from keras.models import Sequential
from keras import layers

model = Sequential()

# The Embedding layer is best understood as a dictionary that maps integer indices
# (which stand for specific words) to dense vectors. It takes integers as input, it looks up
# these integers in an internal dictionary, and it returns the associated vectors. It’s effec-
# tively a dictionary lookup
max_tokens = max_features
embedding_dim = 8

# Add the Embedding layer
# THe output of Embedding layer is 3D tensor of shape (num_samples, max_word_length, vector_dim)
model.add(layers.Embedding(max_tokens, embedding_dim, input_length=maxlen))

model.add(layers.Flatten()) # Here we are loosing the sequence

# Add classifier layer
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss= 'binary_crossentropy', metrics=['acc'])

model.fit(x_train, y_train, epochs= 5, batch_size= 32, validation_split= 0.2)

