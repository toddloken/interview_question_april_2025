import requests
import json

url = 'http://127.0.0.1:5000/predict_basic'
headers = {'Content-Type': 'application/json'}

# List of example requests
examples = [
    {
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "floors": 1.5,
        "sqft_above": 1200,
        "sqft_basement": 600,
        "zipcode": 98178
    },
    {
        "bedrooms": 4,
        "bathrooms": 3,
        "sqft_living": 2500,
        "sqft_lot": 6000,
        "floors": 2,
        "sqft_above": 2000,
        "sqft_basement": 500,
        "zipcode": 98052
    },
    {
        "bedrooms": 2,
        "bathrooms": 1,
        "sqft_living": 900,
        "sqft_lot": 3000,
        "floors": 1,
        "sqft_above": 900,
        "sqft_basement": 0,
        "zipcode": 98103
    },
    {
        "bedrooms": 5,
        "bathrooms": 4,
        "sqft_living": 3500,
        "sqft_lot": 8000,
        "floors": 2.5,
        "sqft_above": 2800,
        "sqft_basement": 700,
        "zipcode": 98006
    },
    {
        "bedrooms": 1,
        "bathrooms": 1,
        "sqft_living": 600,
        "sqft_lot": 2000,
        "floors": 1,
        "sqft_above": 600,
        "sqft_basement": 0,
        "zipcode": 98109
    }
]

# Sending requests and printing responses
for i, example in enumerate(examples):
    response = requests.post(url, data=json.dumps(example), headers=headers)
    print(f"Response for Example {i+1}: {response.json()}")
