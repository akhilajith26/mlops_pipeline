from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from train_requests import TrainModel
from model import train_model
from PIL import Image
import numpy as np
import uvicorn
import boto3
import os

app = FastAPI()


# AWS S3 configuration
S3_BUCKET_NAME = "keras-model-bucket"
S3_MODEL_PATH = "models/my_model.keras"
LOCAL_MODEL_PATH = "models/my_model.keras"

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


# Load the model at startup
model = load_model()

# Load pre-trained model
# model = load_model("models/my_model.keras")


def preprocess_image(image: Image.Image):
    """Preprocess the uploaded image to make it model-compatible."""
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Rescale pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 224, 224, 3)
    return image


@app.get("/")
def read_root():
    return {"message": "Welcome to Car Damage Prediction"}


@app.post("/train")
async def train(train: TrainModel):
    try:
        train_data = train.model_dump(exclude_none=True)
        number_of_epoch = train_data.get("epoch")
        batch_size = train_data.get("batch_size")
        output_size = train_data.get("output_size")
        version = train_data.get("version")
        response_text = train_model(
            epochs=number_of_epoch,
            output_size=output_size,
            batch_size=batch_size,
            version=version,
        )
        return response_text

    except Exception as e:
        return {"error": str(e)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Open and preprocess image
        image = Image.open(file.file)
        image = preprocess_image(image)

        # Perform prediction
        # prediction = model.predict(image)

        # # Since the prediction is an array, extract the first element
        # predicted_value = prediction[0][0]  # Extract the scalar from the array

        # # Interpret the result
        # if predicted_value > 0.5:
        #     result = "Damaged"
        # else:
        #     result = "Undamaged"

        # return {"prediction": result}
        prediction = model.predict(image)

        # Interpret result
        if prediction > 0.5:
            result = "Damaged"
        else:
            result = "Undamaged"

        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
