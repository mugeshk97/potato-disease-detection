from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests

endpoint = 'http://localhost:8601/v1/models/potato-disease/labels/production:predict'

app = FastAPI()

class_names = ['potato_early_blight', 'potato_healthy', 'potato_late_blight']

def read_file(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = read_file(await file.read())
    image_batch = np.expand_dims(image, 0)
    data = {
        'instances': image_batch.tolist()
        }
    response = requests.post(endpoint, json=data)
    prediction = np.array(response.json()["predictions"][0])
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
