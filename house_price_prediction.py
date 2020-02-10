from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print("train_data.shape: ", train_data.shape)

# Here we have to do feature wise normalization because each 13 params are at very different scales
# Mean Normalization
mean = train_data.mean(axis= 0) # Column-wise norm i.e. each feature-wise norm is performed
train_data -= mean
std = train_data.std(axis= 0)
train_data /= std

test_data -= mean
test_data /= std


# Build the model

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(test_data.shape[1], )))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1)) # This is the regression layer

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
    return model


# The K-fold validation
# When we have limited for training and validation.
# Here in this method rather than dividing the training dataset in k partitions. Say k = 5
# We train the network with k-1 partion as training and corresponding 1 partition as validation set.
# Final accuracies is given by taking average of the k training

import numpy as np
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0)

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)