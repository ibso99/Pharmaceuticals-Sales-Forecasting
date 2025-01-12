from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import tensorflow as tf
import os

# Load the trained model and scaler
model_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'lstm_model.pkl')
with open(model_path, 'rb') as f:
    saved_data = pickle.load(f)
model = saved_data['model']
scaler = saved_data['scaler']

# Initialize FastAPI
app = FastAPI()

# Define input data schema using Pydantic
class PredictionRequest(BaseModel):
    recent_sales: list[float]  # List of recent sales values
    days_to_predict: int       # Number of future days to predict

# Define the prediction function
def predict_future_sales(model, scaler, recent_sales_values, days_to_predict=6):
    """
    Predict future sales values using the trained LSTM model.
    """
    # Scale the input data
    recent_sales_scaled = scaler.transform(np.array(recent_sales_values).reshape(-1, 1))
    
    # Reshape the input data into 3D format: [samples, time_steps, features]
    last_5_days_scaled = recent_sales_scaled.reshape(1, -1, 1)  # 1 sample, 5 time steps, 1 feature
    
    predictions = []

    for _ in range(days_to_predict):
        # Predict the next day's sales
        predicted_scaled = model.predict(last_5_days_scaled, verbose=0)
        
        # Inverse transform the prediction
        predicted = float(scaler.inverse_transform(predicted_scaled)[0, 0])
        predictions.append(predicted)
        
        # Update the input for the next prediction
        predicted_scaled = predicted_scaled.reshape(1, 1, 1)  # Reshaping to (1, 1, 1) to match the input format
        last_5_days_scaled = np.append(last_5_days_scaled[:, 1:, :], predicted_scaled, axis=1)
    
    return predictions

# Define the API endpoint for prediction
@app.post("/predict/")
def predict_sales(request: PredictionRequest):
    """
    Predict future sales based on recent sales data and a specified number of days.
    """
    # Validate input length
    if len(request.recent_sales) != 5:  
        raise HTTPException(status_code=400, detail="Recent sales must contain exactly 5 values.")

    # Make predictions
    try:
        predictions = predict_future_sales(
            model=model,
            scaler=scaler,
            recent_sales_values=request.recent_sales,
            days_to_predict=request.days_to_predict
        )
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve the index.html 
from fastapi.responses import FileResponse

@app.get("/")
async def get_index():
    return FileResponse("app/index.html")  
