import base64
import requests
import json



# Change this to your local or deployed URL
API_URL = "http://127.0.0.1:5000/detect"

API_KEY = "hackathon-demo-key-123"

# Path to an MP3 file
AUDIO_FILE = "test.mp3"

with open(AUDIO_FILE, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    API_URL,
    headers={
        "X-API-KEY": API_KEY,
        "Content-Type": "application/json"
    },
    json={
        "audio_base64": audio_b64
    }
)

print(response.status_code)
print(json.dumps(response.json(), indent=2))