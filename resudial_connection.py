from keras import layers

# When the size of Residucal connections are same
x = ....
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.add([y, x]) # The Residual Connnection. In Residual connection the concatenation happens by adding the activation outputs

# When the size of Residual Connections are not same

from keras import layers
xx = ...
y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.MaxPooling2D(2, strides=2)(y)

residual = layers.Conv2D(128, 1, strides=2, padding='same')(x) # Uses a 1 × 1 convolution to linearly downsample the original x tensor to the same shape as y
y = layers.add([y, residual])