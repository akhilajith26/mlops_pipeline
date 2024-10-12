import requests

# Define the API endpoint
url = "http://127.0.0.1:8000/predict"

# Image to upload
files = {"file": open("D:\Scale Up\MlOps_Pipeline\0002.jpg", "rb")}

# Post request to get prediction
response = requests.post(url, files=files)

# Print the response from API
print(response.json())
