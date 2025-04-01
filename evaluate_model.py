import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json

# Load the data
house_data_path = 'data/kc_house_data.csv'
demographics_data_path = 'data/zipcode_demographics.csv'

house_data = pd.read_csv(house_data_path, dtype={'zipcode': str})
demographics = pd.read_csv(demographics_data_path, dtype={'zipcode': str})

# Merge the house data with demographics
data = house_data.merge(demographics, on='zipcode', how='left')

# Load the features used during training
features_path = 'model/model_features.json'
with open(features_path, 'r') as f:
    model_features = json.load(f)

# Define features and target variable
target = 'price'

X = data[model_features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the model
model_path = 'model/model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the performance of the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
