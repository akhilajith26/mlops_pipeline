# import tensorflow as tf
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import TensorBoard
# from data_preprocessing import preprocess_data
# import os


# def create_model(input_shape):
#     # Load ResNet50 as the base model
#     base_model = ResNet50(
#         input_shape=input_shape, include_top=False, weights="imagenet"
#     )
#     x = tf.keras.layers.Flatten()(base_model.output)
#     x = tf.keras.layers.Dense(512, activation="relu")(x)
#     x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

#     model = Model(base_model.input, x)

#     for layer in base_model.layers:
#         layer.trainable = False

#     model.compile(
#         loss="binary_crossentropy",
#         optimizer=RMSprop(learning_rate=1e-4),
#         metrics=["Accuracy", "Precision", "Recall"],
#     )
#     return model


# def train_model(
#     epochs=5, output_size=(224, 224), batch_size=32, version=1, log_dir="logs"
# ):
#     # Preprocess data
#     train_data, val_data = preprocess_data(output_size, batch_size=batch_size)
#     model = create_model(input_shape=(224, 224, 3))

#     # TensorBoard for logging
#     tensorboard = TensorBoard(log_dir=log_dir)

#     # Train the model
#     model.fit(
#         train_data, validation_data=val_data, epochs=epochs, callbacks=[tensorboard]
#     )

#     # Save the trained model
#     model.save(f"models/car_damage_model_{version}.keras")

#     return {"status": "Model trained and saved"}


# # if __name__ == "__main__":
# #     train_model()

import tensorflow as tf
from tensorflow.keras.applications import VGG16  # Use VGG16 instead of ResNet50
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from data_preprocessing import preprocess_data
import os


def create_model(input_shape):
    # Load VGG16 as the base model
    base_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")

    # Add custom layers on top of VGG16
    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(
        x
    )  # Binary classification output

    model = Model(base_model.input, x)

    # Freeze all the layers in VGG16 (so they don't get updated during training)
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(
        loss="binary_crossentropy",
        optimizer=RMSprop(learning_rate=1e-4),
        metrics=["Accuracy", "Precision", "Recall"],
    )

    return model


def train_model(
    epochs=5, output_size=(224, 224), batch_size=32, version=1, log_dir="logs"
):
    # Preprocess data
    train_data, val_data = preprocess_data(output_size, batch_size=batch_size)

    # Create the model using VGG16
    model = create_model(input_shape=(224, 224, 3))

    # TensorBoard for logging
    tensorboard = TensorBoard(log_dir=log_dir)

    # Train the model
    model.fit(
        train_data, validation_data=val_data, epochs=epochs, callbacks=[tensorboard]
    )

    # Save the trained model
    model.save(f"models/car_damage_model_{version}.keras")

    return {"status": "Model trained and saved"}


# Uncomment below to run the training directly if running standalone
# if __name__ == "__main__":
#     train_model()
