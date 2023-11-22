import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from category_encoders import OneHotEncoder
from sklearn import metrics
import joblib
import warnings

# Supress warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('crop_data.csv')

# Features and Target
X_ = df.drop(['label'], axis=1)
y = df['label']

# OHE
encoder = OneHotEncoder(cols=['season', 'country'])
X = encoder.fit_transform(X_)
joblib.dump(encoder, 'ohe.joblib')

# Label Encoder on target
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
le_mapping_names = dict(zip(le.classes_, le.transform(le.classes_)))

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=df['label'], random_state=1)

# LightGBM
model = lgb.LGBMClassifier(verbose=-1)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
metrics.accuracy_score(y_test, y_pred)

# Save model
FILENAME = 'model.pkl'
pickle.dump(model, open(FILENAME, 'wb'))

# Load model
pickled_model = pickle.load(open(FILENAME, 'rb'))
pickled_model.predict(X_test)
metrics.accuracy_score(y_test, y_pred)


# Simulation: Train on outside data
row_to_predict = df.iloc[13:14, :]
row_to_predict.drop(['label'], axis=1, inplace=True)


# Make predictions on the selected row
load_encoder = joblib.load('ohe.joblib')
row_to_predict_ = load_encoder.transform(row_to_predict)

j = model.predict(row_to_predict_)
print(' '.join(j))
