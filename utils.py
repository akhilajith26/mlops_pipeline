import boto3
import os
import tensorflow as tf


# AWS S3 configuration
S3_BUCKET_NAME = "keras-model-bucket"
S3_MODEL_PATH = "models/vgg16_model.keras"
LOCAL_MODEL_PATH = "models/vgg16_model.keras"

# Initialize S3 client
s3 = boto3.client("s3")


def download_model_from_s3():
    """
    Download the model from S3 if not already present locally.
    """
    if not os.path.exists(LOCAL_MODEL_PATH):
        os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
        print(f"Downloading model from S3: {S3_BUCKET_NAME}/{S3_MODEL_PATH}")
        s3.download_file(S3_BUCKET_NAME, S3_MODEL_PATH, LOCAL_MODEL_PATH)
    else:
        print("Model already exists locally.")


# Load the model
def load_model():
    """
    Load the model after downloading it from S3.
    """
    download_model_from_s3()
    model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
    return model
