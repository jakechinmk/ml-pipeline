# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("compare")

# Create input/output pydantic models
input_model = create_model("compare_input", **{'RevolvingUtilizationOfUnsecuredLines': 0.13716208934783936, 'age': 41.0, 'NumberOfTime30-59DaysPastDueNotWorse': 0.0, 'DebtRatio': 0.44299837946891785, 'MonthlyIncome': 9911.0, 'NumberOfOpenCreditLinesAndLoans': 17.0, 'NumberOfTimes90DaysLate': 0.0, 'NumberRealEstateLoansOrLines': 3.0, 'NumberOfTime60-89DaysPastDueNotWorse': 0.0, 'NumberOfDependents': 1.0})
output_model = create_model("compare_output", prediction=0)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
