FROM python:3.9 as builder
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8000
EXPOSE 6006
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
