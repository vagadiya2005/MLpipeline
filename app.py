import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_API_URL = 'http://model-api:5001/predict'  # Using the container name as the host

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        # Forward the request to the model API
        response = requests.post(MODEL_API_URL, json=data)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
