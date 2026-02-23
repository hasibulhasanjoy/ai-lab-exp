import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical
from keras.datasets import mnist, fashion_mnist, cifar10


def build_model(input_shape, numOfClass):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0, 5))
    model.add(Dense(numOfClass, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_tain = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = build_model((28, 28, 1), 10)

model.fit(x_tain, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test)
print("Loss: ", loss)
print("Accuracy: ", acc)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

x_tain = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = build_model((28, 28, 1), 10)

model.fit(x_tain, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test)
print("Loss: ", loss)
print("Accuracy: ", acc)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = build_model((32, 32, 3), 10)

model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test)
print("Loss: ", loss)
print("Accuracy: ", acc)
