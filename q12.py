import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models


def build_model():
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


# without augmentation
train_datagen_no_aug = ImageDataGenerator(rescale=1.0 / 255)

train_generator_no_aug = train_datagen_no_aug.flow_from_directory(
    "dataset/train", target_size=(150, 150), batch_size=32, class_mode="binary"
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

val_generator = val_datagen.flow_from_directory(
    "dataset/val", target_size=(150, 150), batch_size=32, class_mode="binary"
)

model_no_aug = build_model()

history_no_aug = model_no_aug.fit(
    train_generator_no_aug, epochs=20, validation_data=val_generator
)

# with augmentation
train_datagen_aug = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

train_generator_aug = train_datagen_aug.flow_from_directory(
    "dataset/train", target_size=(150, 150), batch_size=32, class_mode="binary"
)

model_aug = build_model()

history_aug = model_aug.fit(
    train_generator_aug, epochs=20, validation_data=val_generator
)

# compare result
plt.plot(history_no_aug.history["val_accuracy"], label="No Aug")
plt.plot(history_aug.history["val_accuracy"], label="With Aug")

plt.legend()
plt.title("Effect of Data Augmentation on Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.show()
