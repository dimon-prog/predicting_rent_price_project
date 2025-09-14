from fastapi import FastAPI
import joblib
from pydantic import BaseModel, Field
import pandas as pd
import torch
from backend.model.model import PricePredictor
from fastapi.staticfiles import StaticFiles
from torch import float32
import numpy as np

scaler_X = joblib.load("backend/model/data/scaler_X.pkl")
scaler_Y = joblib.load("backend/model/data/scaler_Y.pkl")


input_features = scaler_X.n_features_in_
model = PricePredictor(input_features)
model.load_state_dict(torch.load("backend/model/data/model.pth"))
model.eval()

app = FastAPI()


class RentalData(BaseModel):
    area_sqm: float = Field(..., examples=[50.0], description="Area in square meters")
    rooms: int = Field(..., examples=[1], description="number of rooms")
    has_balcony: bool = Field(..., examples=[False], description="has a apartment a balcony")
    has_terrace: bool = Field(..., examples=[False], description="has a apartment a terrace")
    is_furnished: bool = Field(..., examples=[False], description="has the furniture")
    is_social_housing: bool = Field(..., examples=[False], description="is a social house")
    district: int = Field(..., ge=1, le=23, examples=[10], description="district of the house")


@app.post("/predict")
def predict_price(data: RentalData):
    new_apartment_dict = {
        'area_sqm': [data.area_sqm],
        'rooms': [float(data.rooms)],
        'has_balcony': [data.has_balcony],
        'has_terrace': [data.has_terrace],
        "is_furnished": [data.is_furnished],
        "is_social_housing": [data.is_social_housing],
    }
    for i in range(2, 24):
        if i < 10:
            new_apartment_dict[f"district_0{i}"] = [1.0] if data.district == i else [0.0]
        else:
            new_apartment_dict[f"district_{i}"] = [1.0] if data.district == i else [0.0]

    dataset_pred = pd.DataFrame(new_apartment_dict)
    x_input = scaler_X.transform(dataset_pred)
    x_input = torch.tensor(x_input, dtype=float32)
    with torch.no_grad():
        scaled_prediction = model(x_input)

    real_price = scaler_Y.inverse_transform(scaled_prediction.numpy())
    return {"predicted_rent_price": round(np.exp(real_price[0][0]))}


app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
