from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import joblib

# Load the trained model
model_path = "d:/Coding/python course/random_forest_model.pkl"  # Adjust path as necessary
rf_model = joblib.load(model_path)

# Asset classes
asset_classes = ['stocks', 'fd', 'sip', 'bonds_debentures', 'gold_silver', 'cash']

def index(request):
    return render(request, 'portfolio/index.html')

def predict(request):
    if request.method == 'POST':
        # Get form data
        financial_knowledge = int(request.POST['financial_knowledge'])
        age = int(request.POST['age'])
        time_horizon = int(request.POST['time_horizon'])
        risk_appetite = int(request.POST['risk_appetite'])
        comfort_with_fluctuations = int(request.POST['comfort_with_fluctuations'])
        investment_goal = int(request.POST['investment_goal'])

        # Prepare input data
        input_data = {
            'financial_knowledge': [financial_knowledge],
            'age': [age],
            'time_horizon': [time_horizon],
            'risk_appetite': [risk_appetite],
            'comfort_with_fluctuations': [comfort_with_fluctuations],
            'investment_goal': [investment_goal]
        }
        input_df = pd.DataFrame(input_data)

        # Get prediction
        prediction = rf_model.predict(input_df)

        # Map predictions to asset classes
        result = {asset: value for asset, value in zip(asset_classes, prediction[0])}

        return JsonResponse(result)
