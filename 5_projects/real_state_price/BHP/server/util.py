import os

import json
import pickle

import numpy as np

__locations = None
__data_columns = None
__model = None


def get_estimated_price(location, sqft, bath, bhk):
    # Find the index of the 'location' column in the feature matrix X
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    
    # Initialize a numpy array 'x' with zeros, having the same length as the number of columns in X
    x = np.zeros(len(__data_columns))
    
    # Set the values of specific features in 'x'
    x[0] = sqft  # Set the square feet area
    x[1] = bath  # Set the number of bathrooms
    x[2] = bhk   # Set the number of bedrooms/hall-kitchen (BHK)
    
    # If 'location' exists as a column in X (i.e., loc_index is valid)
    if loc_index >= 0:
        x[loc_index] = 1  # Set the value of 'location' column to 1 in 'x'
        
    # Predict the price using the trained linear regression model (lr_clf) and input feature vector [x]
    # Return the predicted price

    return round(__model.predict([x])[0], 2)

def get_location_names():
    return __locations


def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    # Get the directory path where the current script is located
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # load column names from json file using absoulte file path
    columns_file_path = os.path.join(dir_path, "artifacts", "columns.json")
    with open(columns_file_path, "r") as f:
        __data_columns = json.load(f)["data_columns"]
        __locations = __data_columns[3:]

    # Load the trained model from pickle file using absolute file path
    model_file_path = os.path.join(dir_path, "artifacts", "banglore_home_price_model.pickle")
    with open(model_file_path, "rb") as f:
        __model = pickle.load(f)
    
    print("Loading saved artifacts...done")


if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price("Indira Nagar", 1000, 2, 2))