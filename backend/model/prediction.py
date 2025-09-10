import torch
import pandas as pd
import joblib
from model import PricePredictor
from torch import float32

new_apartment_dict = {
    'area_sqm': [100.0],
    'rooms': [1.0],
    'is_first_occupancy': [0.0],
    'is_balkony': [0.0],
    'top_spot': [0.0],
    'new_renovation': [0.0],
    'district_02': [0.0], 'district_03': [1.0], 'district_04': [0.0],
    'district_05': [0.0], 'district_06': [0.0], 'district_07': [0.0],
    'district_08': [0.0], 'district_09': [0.0], 'district_10': [0.0],
    'district_11': [0.0], 'district_12': [0.0], 'district_13': [0.0],
    'district_14': [0.0], 'district_15': [0.0], 'district_16': [0.0],
    'district_17': [0.0], 'district_18': [0.0], 'district_19': [0.0],
    'district_20': [0.0], 'district_21': [0.0], 'district_22': [0.0],
    'district_23': [0.0]
}
dataset_pred = pd.DataFrame(new_apartment_dict)

scaler_X = joblib.load("scaler_X_2.pkl")
scaler_Y = joblib.load("scaler_Y_2.pkl")

X_input = scaler_X.transform(dataset_pred)
X_input = torch.tensor(X_input, dtype=float32)

input_features = scaler_X.n_features_in_
model = PricePredictor(input_features)
model.load_state_dict(torch.load("model_2.pth"))

model.eval()

with torch.no_grad():
    scaled_prediction = model(X_input)

real_price = scaler_Y.inverse_transform(scaled_prediction.numpy())

print(real_price[0][0])