from flask import Flask, request, jsonify
import pickle
import json
import pandas as pd
import uuid
from datetime import datetime
import logging
import os


class PredictionService:
    def __init__(self, default_version='v5', log_level=logging.DEBUG):
        "Class Constructor"
        self.app = Flask(__name__)
        self.default_version = default_version
        self.demographics = pd.read_csv('../data/zipcode_demographics.csv')
        self.setup_routes()
        logging.basicConfig(level=log_level)

    def setup_routes(self):
        """Set up the Flask routes"""
        self.app.route('/predict', methods=['POST'])(self.predict)
        self.app.route('/predict_basic', methods=['POST'])(self.predict_basic)
        self.app.route('/update_model', methods=['POST'])(self.update_model)

    def load_model(self, version):
        """
            Load model from versions
        """
        model_path = f'models/{version}/model.pkl'
        features_path = f'models/{version}/model_features.json'

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(features_path, 'r') as f:
            model_features = json.load(f)

        return model, model_features

    def generate_metadata(self, version, model_features):
        """
        Generate metadata for prediction responses
        """
        return {
            "model": "basic model",
            "version": version,
            "features_used": model_features,
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat()
        }

    def predict(self):
        """Handle predictions using the model"""
        try:
            version = request.args.get('version', self.default_version)
            model, model_features = self.load_model(version)

            data = request.get_json()
            logging.debug(f"Received data: {data}")
            data_df = pd.DataFrame([data])

            # Add demographic data
            data_df = data_df.merge(self.demographics, on='zipcode', how='left')
            logging.debug(f"Data after merging with demographics: {data_df}")

            # Ensure columns are in the same order as during training
            data_df = data_df[model_features]

            prediction = model.predict(data_df)
            logging.debug(f"Prediction: {prediction}")

            # Generate metadata
            metadata = self.generate_metadata(version, model_features)

            response = {
                'prediction': prediction[0],
                'metadata': metadata
            }
            return jsonify(response)
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            return jsonify({'error': str(e)})

    def predict_basic(self):
        """Handle predictions using the basic model features"""
        try:
            version = request.args.get('version', self.default_version)
            logging.debug(f"Version: {version}")
            model, model_features = self.load_model(version)

            basic_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                              'sqft_above', 'sqft_basement', 'zipcode']

            data = request.get_json()
            logging.debug(f"Received data: {data}")
            data_df = pd.DataFrame([data])

            # Add demographic data to ensure all necessary features are included
            data_df = data_df.merge(self.demographics, on='zipcode', how='left')
            logging.debug(f"Data for basic prediction after merging with demographics: {data_df}")

            # Ensure columns are in the same order as required by the basic model
            data_df = data_df[basic_features + list(self.demographics.columns)]
            logging.debug(f"Data for basic prediction: {data_df}")

            # Filter to match the expected features during training
            data_df = data_df[model_features]
            logging.debug(f"Filtered data for prediction: {data_df}")

            prediction = model.predict(data_df)
            logging.debug(f"Prediction: {prediction}")

            # Generate metadata
            metadata = self.generate_metadata(version, model_features)

            response = {
                'prediction': prediction[0],
                'metadata': metadata
            }
            return jsonify(response)
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            return jsonify({'error': str(e)})

    def update_model(self):
        """Handle updating model files"""
        try:
            version = request.args.get('version')
            if not version:
                return jsonify({'error': 'Model version must be specified'}), 400

            model_file = request.files.get('model')
            features_file = request.files.get('features')

            if not model_file or not features_file:
                return jsonify({'error': 'Both model and features files must be uploaded'}), 400

            version_path = f'models/{version}'
            os.makedirs(version_path, exist_ok=True)

            model_file_path = os.path.join(version_path, 'model.pkl')
            features_file_path = os.path.join(version_path, 'model_features.json')

            model_file.save(model_file_path)
            features_file.save(features_file_path)

            # Verify that the files are not empty or corrupted
            with open(model_file_path, 'rb') as f:
                pickle.load(f)  # Attempt to load the model to verify it's valid
            with open(features_file_path, 'r') as f:
                json.load(f)  # Attempt to load the features to verify they're valid

            return jsonify({'message': f'Model version {version} updated successfully'}), 200
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            return jsonify({'error': str(e)}), 500

    def run(self, debug=True):
        self.app.run(debug=debug)

if __name__ == '__main__':
    prediction_service = PredictionService(
        default_version='v5',  # Use a specific model version as default
        log_level=logging.INFO  # Change logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    )
    prediction_service.run(debug=True)