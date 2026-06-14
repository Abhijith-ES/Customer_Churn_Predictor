from fastapi import FastAPI, Request, HTTPException
from src.schemas import CustomerRequestData, PredictionResponseModel
from src.predict import predict_churn
from src.utils import load_random_forest_model, load_preprocessor
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app : FastAPI):
    app.state.model = load_random_forest_model()
    app.state.preprocessor = load_preprocessor()
    yield

app = FastAPI(lifespan=lifespan)

@app.get('/')
def home():
    return {'Message' : 'Customer Churn Predictor Is Running...!'}

@app.get('/health')
def health_check():
    return {'status' : 'Healthy'}

@app.post('/predict', response_model=PredictionResponseModel)
def predict_customer_churn(customer_data: CustomerRequestData, request : Request):
    model = request.app.state.model
    preprocessor = request.app.state.preprocessor

    try:
        prediction_response = predict_churn(customer_data, model, preprocessor)
        return prediction_response
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction Failed due to Some Error.")