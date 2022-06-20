import json
import pickle
from pyexpat import model

from django.shortcuts import redirect
from flask import Flask, request, jsonify, url_for
from Model_files.ml_model import predict_mpg
import requests

##creating a flask app and naming it "app"
app = Flask(__name__)
@app.route('/test', methods=['GET'])
def test():
    return 'Pinging Model Application!!'

@app.route('/predict', methods=['POST'])
def predict():
        vehicle = request.get_json()
        with open('./Model_files/model.bin', 'rb') as f_in:
            model = pickle.load(f_in)
            f_in.close()

        predictions = predict_mpg(vehicle, model)

        result = {
                'mpg_prediction': list(predictions)
            }
        return jsonify(result)

vehicle_config = {
        'cylinders': [8, 6, 8],
        'displacement': [155.0, 160.0, 165.5],
        'horsepower': [93.0, 130.0, 98.0],
        'weight': [2500.0, 3150.0, 2600.0],
        'acceleration': [15.0, 14.0, 16.0],
        'model Year': [81, 80, 78],
        'x0-ASIA': [0.0,0.0,0.0],
        'x0-USA': [0.0,0.0,0.0]
        }


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
    url = "http://0.0.0.0:9696/predict"
    r = requests.post(url, json =  vehicle_config)
    r.text.strip()

#with open('./Model_files/model.bin', 'rb') as f_in:
#    model = pickle.load(f_in)
#    f_in.close()
#x = predict_mpg(vehicle_config, model)
#print(x)