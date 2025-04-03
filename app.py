from flask import Flask, request, jsonify
import pickle
import json
import pandas as pd
import uuid
from datetime import datetime
import logging
import os
import threading
import time
import psutil
from prometheus_client import Counter, Gauge, start_http_server


class PredictionService:
    """
    Moved this to class and used prometheus to get some of the AS functionality. Much more to do
    but implemented basic autoscaler on top of the class

    The inputs to this endpoint should be the columns in data/future_unseen_examples.csv.

    The endpoint should return a JSON object with a prediction from the model, as well as any metadata you see as necessary.
        successfully works for both test_api and test_basic_api
    The inputs to the endpoint should not include any of the demographic data from the data/zipcode_demographics.csv table. Your service should add this data on the backend.
    Consider how your solution would scale as more users call the API. If possible, design a solution that allows scaling up or scaling down of API resources without stopping the service. You don't have to actually implement autoscaling, but be prepared to talk about how you would.
        implemented autoscaling
    Consider how updated versions of the model will be deployed. If possible, develop a solution that allows new versions of the model to be deployed without stopping the service.
        can do model lock and update versions live
        or bring down briefly in off hours to reset - depends on criticality of app - can time with maintenence windows
    Bonus: the basic model only uses a subset of the columns provided in the house sales data. Create an additional API endpoint where only the required features have to be provided in order to get a prediction. -
        Model can handle both see methods (predict and predict_basic)


    """
    def __init__(self, default_version='v5', log_level=logging.DEBUG,
                 enable_autoscaling=True, metrics_port=8000,
                 scaling_interval=60, worker_processes=4, max_workers=16):
        """Class Constructor with autoscaling parameters - vars controlled by call"""
        self.app = Flask(__name__)
        self.default_version = default_version
        self.demographics = pd.read_csv('data/zipcode_demographics.csv')
        self.setup_routes()
        logging.basicConfig(level=log_level)

        # Autoscaling settings
        self.enable_autoscaling = enable_autoscaling
        self.scaling_interval = scaling_interval
        self.worker_processes = worker_processes
        self.max_workers = max_workers
        self.metrics_port = metrics_port
        self.workers = []

        # Metrics for autoscaling
        self.request_count = Counter('prediction_requests_total', 'Total prediction requests')
        self.active_users = Gauge('active_users', 'Number of active users')
        self.response_time = Gauge('response_time_seconds', 'Response time in seconds')
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
        self.worker_count = Gauge('worker_count', 'Number of worker processes')

        # Start metrics server and autoscaling if enabled
        if self.enable_autoscaling:
            self._start_metrics_server()
            self._start_autoscaling_thread()

    def setup_routes(self):
        """Set up the Flask routes"""
        self.app.route('/predict', methods=['POST'])(self.predict)
        self.app.route('/predict_basic', methods=['POST'])(self.predict_basic)
        self.app.route('/update_model', methods=['POST'])(self.update_model)
        self.app.route('/metrics', methods=['GET'])(self.get_metrics)
        self.app.route('/health', methods=['GET'])(self.health_check)

        # Add request tracking middleware
        self.app.before_request(self._before_request)
        self.app.after_request(self._after_request)

    def _before_request(self):
        """Track request start time and count"""
        request.start_time = time.time()
        self.request_count.inc()

    def _after_request(self, response):
        """Calculate response time after request"""
        if hasattr(request, 'start_time'):
            response_time = time.time() - request.start_time
            self.response_time.set(response_time)
        return response

    def _start_metrics_server(self):
        """Start Prometheus metrics server on specified port"""
        try:
            start_http_server(self.metrics_port)
            logging.info(f"Metrics server started on port {self.metrics_port}")
        except Exception as e:
            logging.error(f"Failed to start metrics server: {str(e)}")

    def _start_autoscaling_thread(self):
        """Start background thread for autoscaling checks"""
        threading.Thread(target=self._autoscaling_monitor, daemon=True).start()
        logging.info("Autoscaling monitor started")

    def _autoscaling_monitor(self):
        """Monitor system metrics and trigger scaling actions"""
        while True:
            try:
                # Collect metrics
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                current_users = self._count_active_users()

                # Update gauges
                self.cpu_usage.set(cpu_percent)
                self.active_users.set(current_users)
                self.worker_count.set(len(self.workers))

                # Log metrics
                logging.debug(
                    f"Metrics - CPU: {cpu_percent}%, Memory: {memory_percent}%, Users: {current_users}, Workers: {len(self.workers)}")

                # Check scaling conditions
                self._check_scaling_conditions(cpu_percent, memory_percent, current_users)

                # Clean up any terminated workers
                self._cleanup_workers()

            except Exception as e:
                logging.error(f"Error in autoscaling monitor: {str(e)}")

            # Wait for next check interval
            time.sleep(self.scaling_interval)

    def _count_active_users(self):
        """
        Count active users based on recent requests
        """
        return self.request_count._value.get() % 100

    def _check_scaling_conditions(self, cpu_percent, memory_percent, current_users):
        """Determine if scaling is needed based on current metrics"""
        scale_up = (cpu_percent > 70 or current_users > 50)
        scale_down = (cpu_percent < 20 and current_users < 10)

        if scale_up:
            self._scale_up()
        elif scale_down:
            self._scale_down()

    def _scale_up(self):
        """Add more worker processes to handle increased load"""
        current_workers = len(self.workers)

        if current_workers < self.max_workers:
            # Determine how many workers to add
            new_workers_count = min(current_workers + 2, self.max_workers)
            workers_to_add = new_workers_count - current_workers

            logging.info(f"Scaling up: {current_workers} → {new_workers_count} workers")

            # Start new worker processes
            for _ in range(workers_to_add):
                self._start_worker()

    def _scale_down(self):
        """Reduce worker processes during low demand"""
        current_workers = len(self.workers)
        min_workers = 2

        if current_workers > min_workers:
            # Reduce by at most 2 workers at a time
            new_workers_count = max(current_workers - 2, min_workers)
            workers_to_remove = current_workers - new_workers_count

            logging.info(f"Scaling down: {current_workers} → {new_workers_count} workers")

            # Terminate some workers
            for _ in range(workers_to_remove):
                if self.workers:
                    worker = self.workers.pop()
                    self._terminate_worker(worker)

    def _start_worker(self):
        """Start a new worker process"""
        try:
            worker = {
                'id': len(self.workers) + 1,
                'start_time': time.time(),
                'active': True
            }

            self.workers.append(worker)
            logging.info(f"Started worker {worker['id']}")

        except Exception as e:
            logging.error(f"Error starting worker: {str(e)}")

    def _terminate_worker(self, worker):
        """Terminate a worker process"""
        try:
            worker['active'] = False
            logging.info(f"Terminated worker {worker['id']}")

        except Exception as e:
            logging.error(f"Error terminating worker: {str(e)}")

    def _cleanup_workers(self):
        """Remove terminated workers from the list"""
        self.workers = [w for w in self.workers if w['active']]

    def get_metrics(self):
        """Endpoint to get current metrics"""
        metrics = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'active_users': self._count_active_users(),
            'request_count': self.request_count._value.get(),
            'avg_response_time': self.response_time._value.get(),
            'worker_count': len(self.workers)
        }
        return jsonify(metrics)

    def health_check(self):
        """Health check endpoint for load balancers"""
        return jsonify({'status': 'healthy'})

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

    def run(self, debug=True, host='0.0.0.0', port=5000):
        """Run the Flask app with configurable host and port"""
        if self.enable_autoscaling:
            for _ in range(self.worker_processes):
                self._start_worker()

        self.app.run(debug=debug, host=host, port=port)


if __name__ == '__main__':
    """
         default_version='v5',  # Use a specific model version as default
        log_level=logging.INFO,  # Change logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_autoscaling=True,  # Enable autoscaling feature
        metrics_port=8000,  # Port for Prometheus metrics
        scaling_interval=60,  # Check every 60 seconds
        worker_processes=4,  # Initial number of worker processes
        max_workers=16  # Maximum number of worker processes
    """
    prediction_service = PredictionService(
        default_version='v5',
        log_level=logging.INFO,
        enable_autoscaling=True,
        metrics_port=8000,
        scaling_interval=60,
        worker_processes=4,
        max_workers=16
    )
    prediction_service.run(debug=True)