# Machine Learning Model Deployment

This repository contains two Python scripts for a machine learning model using LightGBM. The first script trains a model on crop data, and the second script deploys the model using Flask for predictions.

## Instructions

### 1. Training the Model

- Install the required Python packages using `pip install -r requirements.txt`.

- Run the script `train_model.py` to train the LightGBM model on crop data. The script also saves the trained model as 'model.pkl' and the One-Hot Encoder as 'ohe.joblib'.

### 2. Model Deployment with Flask

- Run the script `app.py` to start the Flask web server.

- Use the endpoint `/predict` to make predictions by sending a POST request with JSON data.

Example:

```json
{
  "temperature": 25.5,
  "humidity": 75.0,
  "ph": 6.8,
  "water_availability": 180.0,
  "season": "summer",
  "country": "Nigeria"
}
```

### Note
- Ensure that the required packages are installed (pip install -r requirements.txt).

- The Flask app runs on http://127.0.0.1:5000/ by default.

- Adjust file paths and other configurations as needed.