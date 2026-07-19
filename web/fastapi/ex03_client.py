# ex03_client.py
import requests

url = "http://127.0.0.1:8000/stream"
response = requests.get(url, stream=True)

for chunk in response.iter_lines():
    if chunk:
        print(chunk.decode("utf-8"))