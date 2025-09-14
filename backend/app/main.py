from fastapi import FastAPI
import joblib
from pandas.core.arrays.masked import transpose_homogeneous_masked_arrays
from pydantic import BaseModel, Field
import pandas as pd
import torch
from backend.model.model import PricePredictor
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

class RentalData(BaseModel):
    area_sqm: float = Field(..., examples=[50.0], description="Area in square meters")
    rooms: int = Field(..., examples=[1], description="number of rooms")
    has_balcony: bool = Field(..., examples=[False], description="has a apartment a balcony")
    has_terrace: bool = Field(..., examples=[False], description="has a apartment a terrace")
    is_furnished: bool = Field(..., examples=[False], description="has the furniture")
    is_social_housing: bool = Field(..., examples=[False], description="is a social house")
    number_of_district: int = Field(..., ge=1, le=23, examples=[10], description="district of the house")


@app.post("/predict")
async def root(data: RentalData):
    # model = PricePredictor(data)
    # model.load_state_dict(torch.load("model.pth"))
    print(data)
    return {"message": "data"}
