from flask import Flask, render_template, request
import os
import tifffile as tiff
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the U-Net model
model = tf.keras.models.load_model(r'C:\Users\hazem\Desktop\Computer vision\Flask\Unet_model (1).h5')

# Folder to temporarily store uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def normalize_channel(channel):
    return (channel - np.min(channel)) / (np.max(channel) - np.min(channel))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Save the uploaded .tif file
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    # Load the .tif image using tifffile
    image = tiff.imread(input_path)

    # Preprocess the image (resize, normalize if needed)
    input_image = np.expand_dims(image, axis=0)  # Assuming model expects a batch of images

    # Get model prediction
    predicted_mask = model.predict(input_image)

    # Post-process the output
    predicted_mask = np.squeeze(predicted_mask)  # Remove extra batch dimension
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Binarize mask (thresholding)

    # Convert the input image to RGB (assuming image has at least 3 bands)
    red_band = normalize_channel(image[:, :, 3])  # Band 4 - Red
    green_band = normalize_channel(image[:, :, 2])  # Band 3 - Green
    blue_band = normalize_channel(image[:, :, 1])  # Band 2 - Blue

    rgb_image = np.stack((red_band, green_band, blue_band), axis=-1)
    rgb_image = (rgb_image * 255).astype(np.uint8)

    # Create an overlay by converting the mask to RGB and blending with the original image
    mask_rgb = np.zeros_like(rgb_image)
    mask_rgb[:, :, 2] = predicted_mask * 255  # Red mask for the segmentation

    # Overlay with transparency
    overlay_image = Image.blend(Image.fromarray(rgb_image), Image.fromarray(mask_rgb), alpha=0.5)

    # Convert the input image and overlaid image to base64 for displaying on the webpage

    # Convert input image to base64
    img_io = io.BytesIO()
    Image.fromarray(rgb_image).save(img_io, 'PNG')
    img_io.seek(0)
    encoded_input_img = base64.b64encode(img_io.getvalue()).decode('utf-8')

    # Convert overlaid image to base64
    img_io_overlay = io.BytesIO()
    overlay_image.save(img_io_overlay, 'PNG')
    img_io_overlay.seek(0)
    encoded_overlay_img = base64.b64encode(img_io_overlay.getvalue()).decode('utf-8')

    return render_template('index.html', rgb_image=encoded_input_img, output_image=encoded_overlay_img)

if __name__ == '__main__':
    app.run(debug=True)
