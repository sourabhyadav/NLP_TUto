import numpy as np
from keras.datasets import imdb
# Keras imdb dataset consists of 25k sentences. Each sentence is converted in a series of integer where each integer represents a word.

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) # num_words are the top N used words
print("Train data size: ", train_data.shape)
print("Train label size:", train_labels.shape)

print("sample train data:", train_data[0])
print("Sample train label", train_labels[0]) # 0 -> Negative Review, 1 -> Positive Review

# We can convert these series of integers back to english language to understand what was the actual review
word_index = imdb.get_word_index()
reverse_word_index = dict( [(value, key) for (key, value) in word_index.items()] )
decoded_review = " ".join(reverse_word_index.get(i - 3, '?') for i in train_data[0]) # Decodes the review. Note that the indices
# are offset by 3 because 0, 1, and 2 are reserved indices for “padding,” “start of sequence,” and “unknown.”
print("decoded_review: ", decoded_review)

print("sample data size:", len(train_data[0]))
print("decoded review size: ", len(decoded_review))



# Preparing the data: converting the list into tensors
# There are 2 methods to do so:
# 1. One-hot encoding: let the max dimention would be 10000 for each training sample. We would encode 1 at the index the word is present into the training sample i.e. review
# 2. Pad the list of training samples so that they all have equal length. So your input tensor will be of shape (samples, word_indices)

# Here we will try out Method 1:
def one_hot_seq(sequences, dimension=10000):
    result = np.zeros((len(sequences), dimension))

    for i, sequences in enumerate(sequences):
        result[i, sequences] = 1.

    return result

x_train = one_hot_seq(train_data)
x_test = one_hot_seq(test_data)

print("vec train shape: ", x_train.shape)
print("Sample vec: ", x_train[0])

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# define the Model
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Define the traing method properly i.e. loss, optimizer, learning rate and accuracy terms using model.compile API
from keras import optimizers
from keras import losses

# Method 1:
model.compile(optimizer=optimizers.RMSprop(lr = 0.001), loss='binary_crossentropy', metrics=['accuracy'])
# Method2:
#model.compile(optimizer=optimizers.RMSprop(lt=0.001), loss=  losses.binary_crossentropy, metrics= [metrics.binary_accuracy])

# Start the training by providing: training data, training labels, batch size and num_epochs using model.fit API
# Method : Training without validation data
#model.fit(x_train, y_train, batch_size=64, epochs=10)
# Method 2: With valiation data
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    batch_size = 128,
                    epochs= 20,
                    validation_data=(x_val, y_val))

"""
# Visualize the training history. Later we would visualize this with tensorboard
import matplotlib.pyplot as plt

history_dict = history.history
print(history_dict.keys())

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['acc']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
"""

# Once the network is trained you can evaluate performance using model.evaluate API
results = model.evaluate(x_test, y_test)
print("Resutls: ", results)