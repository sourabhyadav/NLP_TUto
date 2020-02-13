from keras import layers
from keras import models

# Imp Note: Each timestep t in the output tensor contains information about timesteps 0
# to t in the input sequence—about the entire past.

# RNN layers returns only the last output sequence
model = models.Sequential()
model.add(layers.Embedding(10000, 32))
model.add(layers.SimpleRNN(32))
model.summary()

# RNN layers Returns the full state sequence
model = models.Sequential()
model.add(layers.Embedding(10000, 32))
model.add(layers.SimpleRNN(32, return_sequences= True))
model.summary()

# Stacking of multiple RNN layers
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.summary()
