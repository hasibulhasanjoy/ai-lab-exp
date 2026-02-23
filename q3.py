import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

x = np.linspace(-10, 10, 1000)

y1 = 5 * x + 10

y2 = 3 * (x**2) + 5 * x + 10

y3 = 4 * (x**3) + 3 * (x**2) + 5 * x + 10

x = x.reshape(-1, 1)
y1 = y1.reshape(-1, 1)
y2 = y2.reshape(-1, 1)
y3 = y3.reshape(-1, 1)


def train_model(x, y, title):
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.3, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42
    )

    model = Sequential()

    if title == "Linear Function: y = 5x + 10":
        model.add(Dense(8, activation="relu", input_shape=(1,)))
    elif title == "Quadratic Function":
        model.add(Dense(16, activation="relu", input_shape=(1,)))
        model.add(Dense(8, activation="relu"))
    else:
        model.add(Dense(32, activation="relu", input_shape=(1,)))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(8, activation="relu"))

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=32,
        verbose=0,
    )

    test_loss, test_mae = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    y_pred = model.predict(x)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label="Original Data", color="blue")
    plt.plot(x, y_pred, color="red", label="predicted data")
    plt.title(title)
    plt.legend()
    plt.show()

    return model


print("Training for y = 5x + 10")
model1 = train_model(x, y1, "Linear Function: y = 5x + 10")

print("Training for y = 3x^2 + 5x + 10")
model2 = train_model(x, y2, "Quadratic Function")

print("Training for y = 4x^3 + 3x^2 + 5x + 10")
model3 = train_model(x, y3, "Cubic Function")
