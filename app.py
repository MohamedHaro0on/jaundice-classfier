# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 02:48:56 2022

@author: Arwa
"""

from __future__ import division, print_function

import os
# model
import pickle
#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

import cv2
import numpy as np
import pandas as pd
# Flask utils
from flask import Flask, request, render_template
from skimage.feature import greycomatrix, greycoprops
from werkzeug.utils import secure_filename

# coding=utf-8

# from keras.applications.imagenet_utils import  decode_predictions

# Define a flask app
app = Flask(__name__)

# Load your trained model
# model = pickle.load(open('RandomForestModel2.pkl', 'rb'))
# model = "E:\Grad_project\Deployment-Deep-Learning-Model-master/RandomForestModel2.pkl"

model = pickle.load(open('RandomForestModel2.pkl', 'rb'))


# model=MODEL_PATH.output_lable()


def model_predict(img_path, model):
    # img =  keras.utils.load_img(img_path, target_size=(40, 40))
    imgage = cv2.imread(img_path)
    imgage = cv2.resize(imgage, (40, 40), interpolation=cv2.INTER_AREA)
    median_image = cv2.medianBlur(imgage, 3)

    # segmentaion
    imgage = cv2.cvtColor(median_image, cv2.COLOR_BGR2RGB)
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = imgage.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)
    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.95)

    k = 9
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)
    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(imgage.shape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    def feature_extractor(dataset):
        image_dataset = pd.DataFrame()
        for image in range(dataset.shape[0]):  # iterate through each file
            # print(image)

            df = pd.DataFrame()  # Temporary data frame to capture information for each loop.
            # Reset dataframe to blank after each loop.

            img = dataset[image, :, :]
            ################################################################
            # START ADDING DATA TO THE DATAFRAME

            # Full image
            # GLCM = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
            GLCM = greycomatrix(img, [1], [0])
            GLCM_Energy = greycoprops(GLCM, 'energy')[0]
            df['Energy'] = GLCM_Energy
            GLCM_corr = greycoprops(GLCM, 'correlation')[0]
            df['Corr'] = GLCM_corr
            GLCM_diss = greycoprops(GLCM, 'dissimilarity')[0]
            df['Diss_sim'] = GLCM_diss
            GLCM_hom = greycoprops(GLCM, 'homogeneity')[0]
            df['Homogen'] = GLCM_hom
            GLCM_contr = greycoprops(GLCM, 'contrast')[0]
            df['Contrast'] = GLCM_contr

            GLCM2 = greycomatrix(img, [3], [0])
            GLCM_Energy2 = greycoprops(GLCM2, 'energy')[0]
            df['Energy2'] = GLCM_Energy2
            GLCM_corr2 = greycoprops(GLCM2, 'correlation')[0]
            df['Corr2'] = GLCM_corr2
            GLCM_diss2 = greycoprops(GLCM2, 'dissimilarity')[0]
            df['Diss_sim2'] = GLCM_diss2
            GLCM_hom2 = greycoprops(GLCM2, 'homogeneity')[0]
            df['Homogen2'] = GLCM_hom2
            GLCM_contr2 = greycoprops(GLCM2, 'contrast')[0]
            df['Contrast2'] = GLCM_contr2

            GLCM3 = greycomatrix(img, [5], [0])
            GLCM_Energy3 = greycoprops(GLCM3, 'energy')[0]
            df['Energy3'] = GLCM_Energy3
            GLCM_corr3 = greycoprops(GLCM3, 'correlation')[0]
            df['Corr3'] = GLCM_corr3
            GLCM_diss3 = greycoprops(GLCM3, 'dissimilarity')[0]
            df['Diss_sim3'] = GLCM_diss3
            GLCM_hom3 = greycoprops(GLCM3, 'homogeneity')[0]
            df['Homogen3'] = GLCM_hom3
            GLCM_contr3 = greycoprops(GLCM3, 'contrast')[0]
            df['Contrast3'] = GLCM_contr3

            # Append features from current image to the dataset
            image_dataset = image_dataset.append(df)

        return image_dataset

    test = feature_extractor(segmented_image)

    # Extract features and reshape to right dimension
    input_img1 = np.array(test)
    input_img = np.expand_dims(input_img1, axis=1)  # Expand dims so the input is (num images, x, y, c)
    input_img_for_RF = np.reshape(input_img, (input_img.shape[0], -1))
    # Predict
    img_prediction = model.predict(input_img_for_RF)
    img_prediction = np.argmax(img_prediction, axis=0)

    def output_lable(img_prediction):
        if img_prediction == 0:
            return "Abnormal"
        else:
            return "Normal"

    preds = output_lable(img_prediction)

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)  # ImageNet Decode
        # result = str(pred_class[0][0][1])  # Convert to string
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)
