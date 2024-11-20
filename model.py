# Import necessary libraries
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the crop recommendation dataset from a CSV file
data = pd.read_csv("./Crop_recommendation.csv") 

# Split the data into features (X) and labels (y)
x = data.iloc[:,:-1] # Features: all columns except the last one
y = data.iloc[:,-1]  # Labels: the last column (crop type)

# Split the data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier()

# Train the model using the training data
model.fit(x_train, y_train)

# Save the trained model to a file using pickle for later use
pickle.dump(model, open("model.pkl", "wb"))