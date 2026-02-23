import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical


def build_model(input_shape, numberOfClasses):
    model = Sequential()

    model.add(Flatten(input_shape=input_shape))

    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))

    model.add(Dense(numberOfClasses, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


# mnist digit
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = build_model((28, 28), 10)
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test)
print("MNIST Test Accuracy:", acc)

# Fashion Mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = build_model((28, 28), 10)
history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2)
loss, acc = model.evaluate(x_test, y_test)
print("Fashion Mnist acc", acc)


# Cifar 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = build_model((32, 32, 3), 10)
history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2)
loss, acc = model.evaluate(x_test, y_test)
print("Fashion Mnist acc", acc)
