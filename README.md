# Water-Segmentation-using-Multispectral-and-optical-Data

# Overview 
Water Segmentation Using U-Net
This repository contains a Jupyter notebook for segmenting water bodies from images using a U-Net model. The project is implemented using TensorFlow and Keras, and the dataset consists of multispectral images and their corresponding labels.

# Contents
    -Data Loading and Preprocessing: The notebook loads the images and labels, resizes them to a uniform size, and normalizes the image data.
    -Model Architecture: A U-Net model is constructed for the segmentation task, a popular architecture in image segmentation tasks.
    -Training: The model is trained on the preprocessed dataset.
    -Evaluation: The trained model is evaluated on validation data to assess its performance.
    -Visualization: The notebook includes visualization of the multispectral image bands, segmentation results, and model predictions.
    -Setup and Requirements

    
# Requirements
    -Python 3.x
    -TensorFlow
    -Keras
    -NumPy
    -Matplotlib
    -tifffile
    -PIL


# Data
The data used in this project includes multispectral images (.tif files) and their corresponding label masks (.png files). The images are normalized and resized to 128x128 pixels for training.

# Acknowledgments

This project is based on the U-Net architecture for image segmentation and leverages multispectral imaging for enhanced accuracy.


# Result 

Model acheive validation accuracy of 89.46% and test accuracy of 92.69%

![image](https://github.com/user-attachments/assets/9cecb574-235e-4725-b682-1fedc505f266)
