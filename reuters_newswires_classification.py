from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000) # Only 10000 top words are used here
print("train_data size: ", train_data.shape)
print("train_labels size: ", train_labels.shape)

query_indx = 1
print("Sample train data: ", test_data[query_indx]) # this contains only integers corresponding to words in the sentence

# Get English conversion of given training sample

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[query_indx]])
print("Sample Train data English: ", decoded_newswire)

print("Sample test label: ", train_labels[query_indx])

# Vectorize the dataset
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print("x_train size: ", x_train.shape)
print("Vectorized Sample Sentence: ", x_train[query_indx])

# Vectorize the Lables.
# Method 1: One-hot Encoding
# Please note here we have 46 classes so we will use one-hot encoding called categorical encoding
def to_one_hot(labels, dimension =46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label]  = 1.
    return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

print("One hot labels size: ", one_hot_test_labels.shape)

# Method 2: Use Keras function built funtion
from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

print("One hot labels size: ", one_hot_test_labels.shape)

# Prepare the Validation data
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]


# Build the model
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train, partial_y_train,
          epochs = 30,
          batch_size= 512,
          validation_data=(x_val, y_val))

# Testing the Trained model on testset
results = model.evaluate(x_test, one_hot_test_labels)
print("Accuracy on testset: ", results)

# Getting results on each sample of testset
predictions = model.predict(x_test)
print("Predicted class after softmax: ", np.argmax(predictions[0]))
