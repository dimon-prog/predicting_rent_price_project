from fastapi import FastAPI
import joblib


app = FastAPI()
model = joblib.load("model.pkl")

@app.get("/")
async def root():
    return {"message": "Hellow world"}

