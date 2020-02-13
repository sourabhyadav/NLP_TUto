from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000
maxlen = 500
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)

print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

# Build the model with Embedding Layer and RNN layer
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.SimpleRNN(32))
model.add(layers.Dense(1, activation= 'sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
              loss= 'binary_crossentropy',
              metrics= ['acc'])

model.fit(input_train, y_train,
          epochs= 10,
          batch_size= 128,
          validation_split= 0.2)

