# Base image with Python and TensorFlow
# FROM tensorflow/tensorflow:2.6.0
FROM python:3.9 as builder


# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose port 8000 (default for FastAPI)
EXPOSE 8000
EXPOSE 6006


# Command to run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
