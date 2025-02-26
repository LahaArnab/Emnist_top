from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import re
import base64

app = Flask(__name__)
model = load_model('mnist.h5')

def preprocess_image(image):
    # Convert to grayscale and resize
    image = image.convert('L').resize((28, 28))
    # Invert colors (MNIST style)
    image = Image.eval(image, lambda x: 255 - x)
    # Convert to array and normalize
    img_array = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get image data from canvas
    image_data = request.json['image'].split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return jsonify({'prediction': int(np.argmax(prediction))})

if __name__ == '__main__':
    app.run(debug=True)