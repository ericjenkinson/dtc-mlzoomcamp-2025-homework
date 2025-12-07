import numpy as np
import onnxruntime as ort
from urllib import request
from PIL import Image
from io import BytesIO
import os

# --- DEBUGGING: LIST FILES ---
print("Current Working Directory:", os.getcwd())
print("Files in this directory:", os.listdir('.'))
# -----------------------------

print("Initializing model...")
# We will use the variable below. If the logs show a different name, we will update this.
model_name = 'hair_classifier_patched.onnx' 

if model_name not in os.listdir('.'):
    print(f"ERROR: {model_name} not found! Check the file list above.")
else:
    print(f"Found {model_name}, loading...")

# Initialize session
session = ort.InferenceSession(model_name, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("Model initialized.")

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess(img):
    x = np.array(img, dtype='float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    x = (x - mean) / std
    x = x.transpose((2, 0, 1))
    return np.expand_dims(x, axis=0).astype(np.float32)

def predict(url):
    img = download_image(url)
    img_prepared = prepare_image(img, (200, 200))
    input_data = preprocess(img_prepared)
    outputs = session.run([output_name], {input_name: input_data})
    return float(outputs[0][0][0])

def lambda_handler(event, context):
    try:
        url = event['url']
        result = predict(url)
        return {'prediction': result}
    except Exception as e:
        return {'error': str(e)}