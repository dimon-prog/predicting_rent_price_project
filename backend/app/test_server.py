# backend/app/test_server.py
from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
def predict_test():
    print("ЗАПРОС НА /predict В ТЕСТОВОМ СЕРВЕРЕ ПОЛУЧЕН!")
    return {"message": "POST request successful on test server"}