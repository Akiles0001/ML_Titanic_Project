from fastapi import FastAPI

from predictor import predict
from schemas import Passenger

app = FastAPI(
    title = "Titanic Surivial API",
    version="1.0"

)

@app.get("/")
def home():

    return {
        "message":"Titanic prediction API is running."
    }

@app.post("/predict")
def predict_survival(passenger: Passenger):
    result = predict(passenger.model_dump())
    return result