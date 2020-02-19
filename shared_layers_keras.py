from keras import layers
from keras import Input
from keras.models import Model

lstm = layers.LSTM(32) # Instantiates a single LSTM layer, once

left_input = Input(shape=(None, 128))
left_output = lstm(left_input)

right_input = Input(shape=(None, 128))
right_output = lstm(right_input) # Building the right branch of the model: when you call an existing layer # instance, you reuse its weights.

merged = layers.concatenate([left_output, right_output], axis=-1)
predictions = layers.Dense(1, activation='sigmoid')(merged)

model = Model([left_input, right_input], predictions)
model.fit([left_data, right_data], targets)