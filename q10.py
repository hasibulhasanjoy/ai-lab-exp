import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

IMG_SIZE = (224, 224)

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# freeze all layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"]
)

history_frozen = model.fit(train_data, validation_data=val_data, epochs=15)

# partial fine tuning
for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[-4:]:  # last conv block
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # smaller LR!
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history_partial = model.fit(train_data, validation_data=val_data, epochs=15)

# full fine tuning
for layer in base_model.layers:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"]
)

history_full = model.fit(train_data, validation_data=val_data, epochs=15)

# plot result
import matplotlib.pyplot as plt

plt.plot(history_frozen.history["val_accuracy"], label="Frozen")
plt.plot(history_partial.history["val_accuracy"], label="Partial FT")
plt.plot(history_full.history["val_accuracy"], label="Full FT")

plt.legend()
plt.title("Fine-Tuning Comparison")
plt.show()
