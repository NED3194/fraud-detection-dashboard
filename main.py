
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the model
model = joblib.load("xgb_model.joblib")

# Define request schema
class Transaction(BaseModel):
    features: list  # Should be a list of floats in the correct order

app = FastAPI()

@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array(transaction.features).reshape(1, -1)
    pred = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]
    return {
        "prediction": int(pred),
        "fraud_probability": float(proba)
    }
