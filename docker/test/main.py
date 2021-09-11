from io import StringIO
from numpy import string_
from starlette.requests import Request
import uvicorn
from fastapi import FastAPI, Form, responses
from pydantic import BaseModel
from ml_utils import load_model, predict_with_test_set
from typing import List
from datetime import datetime
import pandas as pd
import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import os

HACKATHON_ENDPOINT = os.getenv("HACKATHON_ENDPOINT")

# templates = Jinja2Templates(directory="html")

page_break = "\n\n========================================================\n"

# defining the main app
app = FastAPI(title="House Price Predictor", docs_url="/")
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# calling the load_model during startup.
# this will train the model and keep it loaded for prediction.
app.add_event_handler("startup", load_model)

# class which is returned in the response
class QueryOut(BaseModel):
    risk: str

class DataIn(BaseModel):
    ExterQual: str
    GrLivArea: int
    KitchenQual: str
    Neighborhood: str
    FirststFlrSF: int
    TotalBsmtSF: str
    BsmtFinSF1: str
    GarageCars: int
    GarageArea: str
    MSSubClass: str
    MSZoning: str
    LotArea: str
    LotShape: str
    BldgType: str
    OverallCond: str
    Exterior1st: str
    Exterior2nd: str
    YearBuilt: str
    HouseStyle: str
    HeatingQC: str
    FullBath: str
    TotRmsAbvGrd: str
    YrSold: str
    Bedroom: str
    HalfBath: str
  
class DataOut(BaseModel):
    housingPrice: str
class DataAccuracyOut(BaseModel):
    accuracy: str


# not working - need to map strin to keys
# @app.post("/predict_price", response_model=DataOut, status_code=200)
# async def predict_risk(query_data :DataIn):
#     housingPrice = predict(query_data)
#     return {"housingPrice": housingPrice}

@app.post("/get_accuracy_of_model_with_testset", response_model=DataAccuracyOut, status_code=200)
async def get_accuracy_of_model_with_testset():
    housingPrice = predict_with_test_set()
    return {"accuracy": housingPrice}


# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=7777, reload=True)
