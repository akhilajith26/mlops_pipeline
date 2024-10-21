# import os
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.model_selection import train_test_split

# base_dir = "D:\Scale Up\MlOps_Pipeline\data"
# train_dir = os.path.join("data", "training")
# validation_dir = os.path.join("data", "validation")


# def preprocess_data(output_size, batch_size):
#     # ImageDataGenerator for data augmentation and scaling
#     train_data_gen = ImageDataGenerator(
#         rescale=1.0 / 255,
#         validation_split=0.2,  # 20% for validation
#         horizontal_flip=True,
#         rotation_range=10,
#     )
#     validation_data_gen = ImageDataGenerator(
#         rescale=1.0 / 255,
#         validation_split=0.2,  # 20% for validation
#         horizontal_flip=True,
#         rotation_range=10,
#     )

#     # Create train and validation generators
#     train_generator = train_data_gen.flow_from_directory(
#         train_dir,
#         target_size=output_size,
#         batch_size=batch_size,
#         class_mode="binary",
#         subset="training",
#     )

#     validation_generator = validation_data_gen.flow_from_directory(
#         validation_dir,
#         target_size=output_size,
#         batch_size=batch_size,
#         class_mode="binary",
#         subset="validation",
#     )

#     return train_generator, validation_generator


# # if __name__ == "__main__":
# #     data_dir = "data"  # Path to the dataset folder
# #     preprocess_data(data_dir)

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to training and validation directories
train_dir = os.path.join("data", "training")
validation_dir = os.path.join("data", "validation")


def preprocess_data(output_size, batch_size):
    # ImageDataGenerator for data augmentation and scaling
    train_data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,  # 20% for validation
        horizontal_flip=True,
        rotation_range=10,
    )
    validation_data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,  # 20% for validation
        horizontal_flip=True,
        rotation_range=10,
    )

    # Create train and validation generators
    train_generator = train_data_gen.flow_from_directory(
        train_dir,
        target_size=output_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",  # Subset for training
    )

    validation_generator = validation_data_gen.flow_from_directory(
        validation_dir,
        target_size=output_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",  # Subset for validation
    )

    return train_generator, validation_generator


# Uncomment below to test data preprocessing directly if running standalone
# if __name__ == "__main__":
#     preprocess_data((224, 224), batch_size=32)
