import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 15

train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    "dataset/train", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    "dataset/val", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
)


def build_model(activation_function, loss_function):

    model = models.Sequential(
        [
            layers.Conv2D(
                32, (3, 3), activation=activation_function, input_shape=(150, 150, 3)
            ),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation=activation_function),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation=activation_function),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation=activation_function),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001), loss=loss_function, metrics=["accuracy"]
    )

    return model


activations = ["relu", "tanh", "sigmoid"]
histories_activation = {}

for act in activations:
    print(f"\nTraining with activation: {act}")
    model = build_model(activation_function=act, loss_function="binary_crossentropy")

    history = model.fit(
        train_generator, epochs=EPOCHS, validation_data=val_generator, verbose=1
    )

    histories_activation[act] = history


losses = ["binary_crossentropy", "mse"]
histories_loss = {}

for loss in losses:
    print(f"\nTraining with loss: {loss}")
    model = build_model(activation_function="relu", loss_function=loss)

    history = model.fit(
        train_generator, epochs=EPOCHS, validation_data=val_generator, verbose=1
    )

    histories_loss[loss] = history


plt.figure(figsize=(10, 6))

for act in activations:
    plt.plot(
        histories_activation[act].history["val_accuracy"], label=f"Activation: {act}"
    )

plt.title("Effect of Activation Functions (Validation Accuracy)")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))

for loss in losses:
    plt.plot(histories_loss[loss].history["val_accuracy"], label=f"Loss: {loss}")

plt.title("Effect of Loss Functions (Validation Accuracy)")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.show()
