import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape the input data to be a 1-dimensional array
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Normalize the input data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# One-hot encode the output data
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the neural network model
model = Sequential()
model.add(Dense(50, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Evaluate the model
scores = model.evaluate(X_test, y_test)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])
