# TL:DR;
# Author: Tunde Adepitan
# Project Details: 
# This project is a machine learning-powered web application 
# that predicts the most suitable crop to plant based on soil and weather data. 
# The app is built using Flask and a Random Forest Classifier.

# --- START HERE ---- >

# Import necessary libraries
import numpy as np 
from flask import Flask, request, render_template
import pickle

# Initialize the Flask application
flask_app = Flask(__name__)

# Load the pre-trained machine learning model using pickle
model = pickle.load(open("model.pkl","rb"))

# Define route for the homepage (index)
@flask_app.route("/")
def Home():
    # Render the index.html template (main page)
    return render_template("index.html")

# Define route for the ML prediction feature
@flask_app.route("/predict", methods=["POST"])
def predict():
    # Get the values from the input form and convert them to floats
    float_feature = [float(x) for x in request.form.values()]

    # Convert the input features into a numpy array for prediction
    features = [np.array(float_feature)]

    # Use the loaded model to make a prediction based on the input features
    prediction = model.predict(features)

    # Render the index page again, displaying the prediction result
    return render_template("index.html", prediction_text="The Predicted Crop is {}".format(prediction))

# Run the Flask app in debug mode
if __name__ == "__main__":
    flask_app.run(debug=True)
