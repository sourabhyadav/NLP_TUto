# Getting dataset form Keras
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Check the correctness o the shape
print("train size: ", train_images.shape, " train lable size: ", train_labels)
print("test size: ", test_images.shape, " test label size: ", test_labels)

# Defien the network arch
from keras import models
from keras import layers

# The Forward Pass
input_tensor = layers.Input(shape=(784,))
x = layers.Dense(32, activation='relu')(input_tensor)
out_tensor = layers.Dense(10, activation='softmax')(x)

net  = models.Model(inputs=input_tensor, outputs=out_tensor)

# The Backward pass
net.compile(optimizer='rmsprop', loss= 'categorical_crossentropy', metrics=['accuracy']) # Add stuf for backpropagation

# Prepare the train data
train_images = train_images.reshape(60000, 28*28)
train_images = train_images.astype('float32')/255

# Prepae the labels
from keras .utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# The Train command in Keras: which does the forward and backwad passes
net.fit(train_images, train_labels, epochs=5, batch_size=128)
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
