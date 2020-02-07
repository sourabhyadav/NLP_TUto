# Getting dataset form Keras
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Check the correctness o the shape
print("train size: ", train_images.shape, " train lable size: ", train_labels)
print("test size: ", test_images.shape, " test label size: ", test_labels)

# Defien the network arch
from keras import models
from keras import layers
from keras import optimizers

# The Forward Pass
net = models.Sequential() # Feed forward network model class
net.add(layers.Dense(612, activation= 'relu', input_shape=(28 * 28, ))) # FC or Dense layer with input dim = 28*28, NA and output dim = 512. Activiation fo each neuron is relu.
net.add(layers.Dense(32, activation='relu')) # Even if you dont provide input_zie it will automatically take the above layer's output as input
net.add(layers.Dense(10, activation = 'softmax')) # Keras automatically takes input dim as 512 an out put dim = 10, As this is last layer we add activation = 'softmax'

# The Backward pass
# Below are two ways you can pass optimizers
# Method 1:
#net.compile(optimizer='rmsprop', loss= 'categorical_crossentropy', metrics=['accuracy']) # Add stuf for backpropagation
# Method 2:
net.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mse', metrics=['accuracy'])

# Prepare the train data
train_images = train_images.reshape(60000, 28*28)
train_images = train_images.astype('float32')/255

# Prepae the labels
from keras .utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# The Train command in Keras: which does the forward and backwad passes
net.fit(train_images, train_labels, epochs=10, batch_size=128)
# note about epochs and batch_size
"""
Given training data size: 60000
Number of times the network will see complete dataset (aka EPOCH) = 5
In BackProp it will update the gradients at each iterations. Thus, number of gradient updates = Number of iterations
How many iterations it will run? 
Total iterations for 1 epoch = 60000/128 = 468
Total iterations for 5 epoch = 468 * 5 = 2340
Thus total weight updates = 2340 times
"""

# Prepare the test data
test_images = test_images.reshape(10000, 28*28)
test_images = test_images.astype('float32')/255

# The testing command in keras: This will test the trained model and provide the accuracy
test_loss, test_acc = net.evaluate(test_images, test_labels)
print("test_acc: ", test_acc)
