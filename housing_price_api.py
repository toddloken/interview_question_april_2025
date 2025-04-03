import requests
import json
import pandas as pd
import time
import os
import argparse


class HousingPriceApiTester:
    """Class for testing the housing price prediction API with CSV data"""

    def __init__(self, csv_path=None, api_url='http://127.0.0.1:5000/predict'):
        """Initialize the API tester with the CSV path and API URL

        Args:
            csv_path (str, optional): Path to the CSV file with test samples
            api_url (str, optional): API endpoint URL. Defaults to 'http://127.0.0.1:5000/predict'
        """
        self.api_url = api_url
        self.headers = {'Content-Type': 'application/json'}
        self.csv_path = csv_path
        self.data = None
        self.results = []
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_requests = 0
        self.execution_time = 0

    def find_csv_file(self):
        """Find the CSV file containing test samples"""
        # If CSV path is already provided and exists, use it
        if self.csv_path and os.path.exists(self.csv_path):
            print(f"Using CSV file from provided path: {self.csv_path}")
            return self.csv_path

        # Determine the correct path to the CSV file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(script_dir)) if os.path.basename(
            script_dir) == 'data' else script_dir
        default_csv_path = "future_unseen_examples.csv"

        # Try alternative paths if the first one doesn't work
        alternative_paths = [
            default_csv_path,
            os.path.join(script_dir, 'future_unseen_samples.csv'),  # Same directory as script
            'future_unseen_samples.csv',  # Current working directory
            os.path.join(os.getcwd(), 'data', 'future_unseen_samples.csv')  # data folder in current directory
        ]

        for path in alternative_paths:
            try:
                print(f"Trying to load from: {path}")
                if os.path.exists(path):
                    print(f"Successfully located CSV at: {path}")
                    return path
            except Exception as e:
                print(f"Could not access {path}: {e}")

        return None

    def load_data(self):
        """Load and prepare data from the CSV file"""
        path = self.find_csv_file()

        if not path:
            print("Error: Could not find 'future_unseen_samples.csv' in any of the expected locations.")
            print("Please specify the full path to the CSV file when initializing the tester.")
            return False

        try:
            self.data = pd.read_csv(path)
            print(f"Successfully loaded {len(self.data)} samples from: {path}")

            # Convert NaN values to 0 (or appropriate defaults)
            self.data = self.data.fillna(0)
            self.total_requests = len(self.data)
            return True
        except Exception as e:
            print(f"Failed to load data from {path}: {e}")
            return False

    def send_request(self, sample_data):
        """Send a single request to the API

        Args:
            sample_data (dict): Sample data to send to the API

        Returns:
            tuple: (success, response_data, status_code)
        """
        try:
            response = requests.post(
                self.api_url,
                data=json.dumps(sample_data),
                headers=self.headers
            )

            if response.status_code == 200:
                return True, response.json(), response.status_code
            else:
                return False, response.text, response.status_code

        except Exception as e:
            return False, str(e), None

    def process_results(self, sample_id, success, response_data, status_code):
        """Process and store results from an API request

        Args:
            sample_id (int): ID of the sample
            success (bool): Whether the request was successful
            response_data (dict or str): Response data or error message
            status_code (int): HTTP status code
        """
        if success:
            print(f"Sample {sample_id + 1}: Success - Predicted price: ${response_data.get('prediction', 'N/A')}")
            self.results.append({
                "sample_id": sample_id + 1,
                "success": True,
                "prediction": response_data.get('prediction'),
                "status_code": status_code
            })
            self.successful_requests += 1
        else:
            print(f"Sample {sample_id + 1}: Failed - Status code: {status_code}, Response: {response_data}")
            self.results.append({
                "sample_id": sample_id + 1,
                "success": False,
                "error": response_data,
                "status_code": status_code
            })
            self.failed_requests += 1

    def run_tests(self, delay=0.1):
        """Run tests for all samples in the dataset

        Args:
            delay (float, optional): Delay between requests in seconds. Defaults to 0.1.
        """
        if not self.data is not None:
            if not self.load_data():
                return False

        # Reset counters
        self.successful_requests = 0
        self.failed_requests = 0
        self.results = []

        print(f"Testing {self.total_requests} samples...")
        start_time = time.time()

        for i, row in self.data.iterrows():
            # Convert row to dictionary and ensure all values have the right type
            sample = row.to_dict()

            # Send request and process results
            success, response_data, status_code = self.send_request(sample)
            self.process_results(i, success, response_data, status_code)

            # Optional: Add a small delay to avoid overwhelming the server
            time.sleep(delay)

        end_time = time.time()
        self.execution_time = end_time - start_time
        return True

    def save_results(self, filename='prediction_results.csv'):
        """Save results to a CSV file

        Args:
            filename (str, optional): Path to save results. Defaults to 'prediction_results.csv'.
        """
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(filename, index=False)
        print(f"Results saved to '{filename}'")

    def print_summary(self):
        """Print a summary of the test results"""
        print("\n--- SUMMARY ---")
        print(f"Total samples tested: {self.total_requests}")
        print(
            f"Successful predictions: {self.successful_requests} ({self.successful_requests / self.total_requests * 100:.2f}%)")
        print(f"Failed predictions: {self.failed_requests} ({self.failed_requests / self.total_requests * 100:.2f}%)")
        print(f"Execution time: {self.execution_time:.2f} seconds")


def main():
    """Main function to run the API tester from command line"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test housing price prediction API with CSV data')
    parser.add_argument('--csv-path', type=str, help='Path to the CSV file with test samples')
    parser.add_argument('--url', type=str, default='http://127.0.0.1:5000/predict',
                        help='API endpoint URL (default: http://127.0.0.1:5000/predict)')
    args = parser.parse_args()

    # Create and run the tester
    tester = HousingPriceApiTester(csv_path=args.csv_path, api_url=args.url)
    if tester.run_tests():
        tester.print_summary()
        tester.save_results()


if __name__ == "__main__":
    main()