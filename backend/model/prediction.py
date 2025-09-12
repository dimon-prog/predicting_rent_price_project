import torch
import pandas as pd
import joblib
from model import PricePredictor
from torch import float32
import numpy as np


new_apartment_dict = {
    'area_sqm': [60.0],
    'rooms': [2.0],
    'is_new_building': [0.0],
    'has_balcony': [0.0],
    'has_terrace': [0.0],
    "is_furnished": [0.0],
    "is_social_housing": [0.0],
    'district_02': [0.0], 'district_03': [0.0], 'district_04': [0.0],
    'district_05': [0.0], 'district_06': [0.0], 'district_07': [0.0],
    'district_08': [0.0], 'district_09': [0.0], 'district_10': [0.0],
    'district_11': [0.0], 'district_12': [0.0], 'district_13': [0.0],
    'district_14': [0.0], 'district_15': [0.0], 'district_16': [0.0],
    'district_17': [0.0], 'district_18': [0.0], 'district_19': [1.0],
    'district_20': [0.0], 'district_21': [0.0], 'district_22': [0.0],
    'district_23': [0.0]
}
dataset_pred = pd.DataFrame(new_apartment_dict)

scaler_X = joblib.load("data/scaler_X.pkl")
scaler_Y = joblib.load("data/scaler_Y.pkl")

X_input = scaler_X.transform(dataset_pred)
X_input = torch.tensor(X_input, dtype=float32)

input_features = scaler_X.n_features_in_
model = PricePredictor(input_features)
model.load_state_dict(torch.load("data/model.pth"))

model.eval()

with torch.no_grad():
    scaled_prediction = model(X_input)

real_price = scaler_Y.inverse_transform(scaled_prediction.numpy() )

print(np.exp(real_price[0][0]))