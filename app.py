import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
import pickle, joblib

# Create Flask app
app = Flask(__name__)

# Load model and ohe
model = pickle.load(open('model.pkl', 'rb'))
ohe = joblib.load(open('ohe.joblib', 'rb'))


@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    hot_encoder = ohe.transform(query_df)
    prediction = model.predict(hot_encoder)
    return jsonify({'Prediction': str(prediction)})

if __name__ == '__main__':
    app.run(debug=True)