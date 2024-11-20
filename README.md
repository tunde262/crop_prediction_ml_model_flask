# Crop Recommendation System  

This project is a machine learning-powered web application that predicts the most suitable crop to plant based on soil and weather data. The app is built using Flask and a Random Forest Classifier.  

## Project Goal

The goal of the project was to implement a machine learning solution to solve a real-world problem in agriculture.

## Features  
- Predicts crop recommendations using a trained Random Forest Classifier.  
- Accepts key agricultural inputs like Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.  
- Displays the predicted crop on the web interface.  
- Provides an example of integrating machine learning models with web applications.  

## Technologies  
- **Backend:** Flask, Python  
- **Machine Learning:** Random Forest Classifier, Scikit-learn  
- **Frontend:** HTML/CSS  
- **Data Handling:** Pandas, NumPy  
- **Model Deployment:** Pickle  

## Dataset  
The model is trained on the **Crop_recommendation.csv** dataset, which contains data on:  
- Soil nutrients (Nitrogen, Phosphorus, Potassium)  
- Weather factors (Temperature, Humidity, Rainfall)  
- Soil pH levels  
- Corresponding crop labels  

## Installation
Before running the application, install the required Python libraries:  

```bash
pip install flask numpy pandas scikit-learn
```
## How To Run

1. Clone this repository.

2. Ensure the dataset file `Crop_recommendation.csv` is in the project directory.

3. Train the machine learning model by running `model.py`. This will generate the `model.pkl` file.

   ```bash
   python model.py

4. Start the Flask app by running `app.py`:

  ```bash
   python app.py
  ```

5. Open a web browser and navigate to **http://127.0.0.1:5000/**.

6. Enter your sample inputs and click "Predict" to see the recommended crop.

## Acknowledgments

This project was inspired by [AlgoChat](https://youtu.be/Rynv-ueHzwU?si=oQXDLpS_r36D0sjf) for the original concept and dataset.
