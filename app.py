from fastapi import FastAPI, File, UploadFile, Request, Response
from tensorflow.keras.models import load_model
from train_requests import TrainModel
from fastapi.exceptions import HTTPException
from utils import load_model
from model import train_model
from logger import setup_logger
from PIL import Image
import numpy as np
import uvicorn


app = FastAPI()

# Setup logger
logger = setup_logger("fastapi")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Log request details
    logger.info(f"Request URL: {request.url}")
    logger.info(f"Request Method: {request.method}")
    logger.info(f"Request Headers: {request.headers}")
    # logger.info(f"Request Body: {await request.body()}")

    # Process the request and get the response
    try:
        response: Response = await call_next(request)

        # Log response details
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response Headers: {response.headers}")

        # Optionally log the response body (ensure it doesn't exceed a reasonable size)
        if response.status_code == 200 and response.media_type == "application/json":
            response_body = await response.body()
            logger.info(f"Response Body: {response_body.decode()}")

        return response

    except HTTPException as exc:
        # Log exception details
        logger.error(f"HTTP Exception: {exc.detail}")
        raise exc

    except Exception as e:
        # Log any unexpected errors
        logger.error(f"Unhandled Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Load the model at startup
model = load_model()


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
