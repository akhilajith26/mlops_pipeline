# MLOps Pipeline Project

This project demonstrates an end-to-end machine learning pipeline for a computer vision problem, specifically classifying images of damaged and undamaged cars. The model is built using TensorFlow, containerized using Docker, and ready for deployment. The pipeline automates data preprocessing, model training, logging, and model serving via a FastAPI-based REST API.

## Project Structure

```bash
MLOPS_PIPELINE/
│
├── __pycache__/                 # Python cache files
├── .github/                     # GitHub workflows and configurations
├── data/                        # Directory containing training and validation datasets
├── logs/                        # Logs directory for TensorBoard and model training logs
├── models/                      # Trained model storage directory
├── venv/                        # Python virtual environment (optional)
│
├── .dockerignore                # Files to ignore when building Docker images
├── .gitignore                   # Git ignore configuration
├── Dockerfile                   # Dockerfile for containerizing the application
├── docker-compose.yml           # Docker Compose file for multi-container setup
│
├── app.py                       # FastAPI app with the REST API for model serving
├── data_preprocessing.py         # Data preprocessing pipeline (image augmentation, validation split)
├── logger.py                    # Logger configuration file for API and model logging
├── model.py                     # Model architecture and training pipeline (MobileNetV2)
├── mlops_pipeline.ipynb         # Jupyter notebook with additional MLOps experimentation
├── requirements.txt             # Python dependencies for the project
├── test.py                      # Test file for testing the API (model inference)
├── train_requests.py            # Helper script for making API requests for training
├── utils.py                     # Utility functions for the project
├── render.yaml                  # Configuration for cloud rendering or deployment
└── api.log                      # Log file storing API request logs
```

## Key Components

- **`app.py`**: Contains the FastAPI app that serves the model via a REST API. It accepts an image and returns whether the car is damaged or not.
- **`data_preprocessing.py`**: Handles image preprocessing using Keras `ImageDataGenerator` for data augmentation and splitting data into training and validation sets.
- **`model.py`**: Defines the CNN model using MobileNetV2, and includes the training and model saving functions.
- **`logger.py`**: Custom logging configuration to capture error logs and request details from the API.
- **`docker-compose.yml`**: Used to manage Docker services including the API and TensorBoard.
- **`Dockerfile`**: Docker configuration for containerizing the FastAPI application and model inference.
- **`mlops_pipeline.ipynb`**: Jupyter notebook for additional experimentation and debugging.

## Instructions

#### 1. Set Up Environment

Create a virtual environment (optional) and install dependencies:

```bash
python3 -m venv venv
venv\Scripts\activate  # On Mac:  source venv/bin/activate
pip install -r requirements.txt
```

#### 2. Running the Application

To run the application locally using Docker:

```bash
docker-compose up --build
```

This will start both the FastAPI application and TensorBoard for model logs.

#### 3. Training the Model

To train the model:

1. Ensure your data is in the `data/` folder structured with training and validation directories.
2. Run the training process:

   ```bash
   python model.py
   ```

#### 4. API Requests

You can test the API by sending a request to the `/predict` endpoint with an image file:

```bash
curl -X POST "http://localhost:8000/predict" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "file=@path_to_your_image.jpg"
```

#### 5. Logging

Logs are written to `api.log` for API requests and `logs/` for TensorBoard.

#### 6. Testing

Use the `test.py` script to perform unit tests for the API and verify the model’s predictions.

### CI/CD and Deployment

- CI/CD is configured using GitHub Actions (in `.github`).
- Docker images are pushed to Docker Hub.
- The Docker image can be deployed to any container registry and run on cloud platforms like AWS or Google Cloud.
