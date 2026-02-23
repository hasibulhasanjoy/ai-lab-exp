import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models


def build_model(use_dropout=False):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))

    if use_dropout:
        model.add(layers.Dropout(0.5))  # <-- Dropout layer

    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


train_no_aug = ImageDataGenerator(rescale=1.0 / 255)

train_gen_no_aug = train_no_aug.flow_from_directory(
    "dataset/train", target_size=(150, 150), batch_size=32, class_mode="binary"
)

val_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    "dataset/val", target_size=(150, 150), batch_size=32, class_mode="binary"
)

train_aug = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

train_gen_aug = train_aug.flow_from_directory(
    "dataset/train", target_size=(150, 150), batch_size=32, class_mode="binary"
)

model_1 = build_model(use_dropout=False)

history_1 = model_1.fit(train_gen_no_aug, epochs=20, validation_data=val_gen)

model_2 = build_model(use_dropout=True)

history_2 = model_2.fit(train_gen_no_aug, epochs=20, validation_data=val_gen)

model_3 = build_model(use_dropout=False)

history_3 = model_3.fit(train_gen_aug, epochs=20, validation_data=val_gen)

model_4 = build_model(use_dropout=True)

history_4 = model_4.fit(train_gen_aug, epochs=20, validation_data=val_gen)

plt.figure(figsize=(10, 6))

plt.plot(history_1.history["val_accuracy"], label="No Reg")
plt.plot(history_2.history["val_accuracy"], label="Dropout")
plt.plot(history_3.history["val_accuracy"], label="Augmentation")
plt.plot(history_4.history["val_accuracy"], label="Dropout + Aug")

plt.title("Effect of Dropout & Data Augmentation on Overfitting")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.show()
