
import requests
import json

url = 'http://127.0.0.1:5000/predict'
headers = {'Content-Type': 'application/json'}

# List of example requests
examples = [
    {
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "floors": 1.5,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,
        "sqft_above": 1200,
        "sqft_basement": 600,
        "yr_built": 1980,
        "yr_renovated": 0,
        "zipcode": 98178,
        "lat": 47.5112,
        "long": -122.257,
        "sqft_living15": 1340,
        "sqft_lot15": 5650
    }
]

# Sending requests and printing responses
for i, example in enumerate(examples):
    response = requests.post(url, data=json.dumps(example), headers=headers)
    print(f"Response for Example {i+1}: {response.json()}")
