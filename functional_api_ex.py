from keras.models import Sequential, Model
from keras import layers
from keras import Input

# Build a simple Sequential model
model = Sequential()
model.add(layers.Dense(32, activation='relu', input_shape= (64,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

# Equivivalent Functional API
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

model_fun = Model(input_tensor, output_tensor)
model_fun.summary()

# Compiling the and training the model remains same
model_fun.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# Prepare dummy data
import numpy as np
x_train = np.random.random((1000,64))
y_train = np.random.random((1000, 10))

model_fun.fit(x_train, y_train,
          epochs= 5,
          batch_size= 32)
score = model_fun.evaluate(x_train, y_train)
