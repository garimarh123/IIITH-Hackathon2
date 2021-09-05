from io import StringIO
from numpy import string_
from starlette.requests import Request
import uvicorn
from fastapi import FastAPI, Form, responses
from pydantic import BaseModel
from ml_utils import load_model, predict, getData
from typing import List
from datetime import datetime
import pandas as pd
import logging
import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="html")
logging.basicConfig(
    filename="debug.log",
    format="main:%(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    encoding="utf-8",
    level=logging.DEBUG,
)
page_break = "\n\n========================================================\n"

# defining the main app
app = FastAPI(title="Credit Risk Predictor", docs_url="/")
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


@app.get("/predict")
def render_html(request: Request):
    return templates.TemplateResponse(
        "predict_housing_price_html.html", {"request": request}
    )


@app.get("/train")
def render_html(request: Request):
    return templates.TemplateResponse("train_html.html", {"request": request})


@app.get("/home")
def render_html(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/load_model")
def render_html(request: Request):
    accuracy, model = load_model()
    return templates.TemplateResponse(
        "train_html.html", {"request": request, "accuracy": accuracy, "model": model}
    )


@app.post("/predict_risk", response_model=QueryOut, status_code=200)
async def predict_risk(
    request: Request,
    ExterQual: str = Form(...),
    GrLivArea: int = Form(...),
    KitchenQual: str = Form(...),
    Neighborhood: str = Form(...),
    FirststFlrSF: int = Form(...),
    TotalBsmtSF: str = Form(...),
    BsmtFinSF1: str = Form(...),
    GarageCars: int = Form(...),
    GarageArea: str = Form(...),
    MSSubClass: str = Form(...),
    MSZoning: str = Form(...),
    LotArea: str = Form(...),
    LotShape: str = Form(...),
    BldgType: str = Form(...),
    OverallCond: str = Form(...),
    Exterior1st: str = Form(...),
    Exterior2nd: str = Form(...),
    YearBuilt: str = Form(...),
    HouseStyle: str = Form(...),
    HeatingQC: str = Form(...),
    FullBath: str = Form(...),
    TotRmsAbvGrd: str = Form(...),
    YrSold: str = Form(...),
    Bedroom: str = Form(...),
    HalfBath: str = Form(...),
):

    x = list()
    x.append(ExterQual)
    x.append(GrLivArea)
    x.append(KitchenQual)
    x.append(Neighborhood)
    x.append(FirststFlrSF)
    x.append(TotalBsmtSF)
    x.append(BsmtFinSF1)
    x.append(GarageCars)
    x.append(GarageArea)
    x.append(MSSubClass)
    x.append(MSZoning)
    x.append(LotArea)
    x.append(LotShape)
    x.append(BldgType)
    x.append(OverallCond)
    x.append(Exterior1st)
    x.append(Exterior2nd)
    x.append(YearBuilt)
    x.append(HouseStyle)
    x.append(HeatingQC)

    x.append(FullBath)
    x.append(HalfBath)
    x.append(Bedroom)
    x.append(YrSold)
    x.append(TotRmsAbvGrd)

    housingPrice = predict(x)
    return templates.TemplateResponse(
        "predict_housing_price_html.html",
        {"request": request, "housingPrice": housingPrice},
    )


@app.post("/feedback_loop", status_code=200)
def feedback_loop(
    request: Request,
    ExterQual: str = Form(...),
    GrLivArea: int = Form(...),
    KitchenQual: str = Form(...),
    Neighborhood: str = Form(...),
    FirststFlrSF: int = Form(...),
    TotalBsmtSF: str = Form(...),
    BsmtFinSF1: str = Form(...),
    GarageCars: int = Form(...),
    GarageArea: str = Form(...),
    MSSubClass: str = Form(...),
    MSZoning: str = Form(...),
    LotArea: str = Form(...),
    LotShape: str = Form(...),
    BldgType: str = Form(...),
    OverallCond: str = Form(...),
    Exterior1st: str = Form(...),
    Exterior2nd: str = Form(...),
    YearBuilt: str = Form(...),
    HouseStyle: str = Form(...),
    HeatingQC: str = Form(...),
    FullBath: str = Form(...),
    TotRmsAbvGrd: str = Form(...),
    YrSold: str = Form(...),
    Bedroom: str = Form(...),
    HalfBath: str = Form(...),
    housingPrice: str = Form(...),
):
    # indexes = []
    # dataSorted = []
    # for index in range(0, len(data)):
    #     dataSorted.append(list(data[index].dict().values()))
    #     indexes.append(index)
    # dataSorted = pd.DataFrame(dataSorted, index=indexes)

    x = list()
    x.append(ExterQual)
    x.append(GrLivArea)
    x.append(KitchenQual)
    x.append(Neighborhood)
    x.append(FirststFlrSF)
    x.append(TotalBsmtSF)
    x.append(BsmtFinSF1)
    x.append(GarageCars)
    x.append(GarageArea)
    x.append(MSSubClass)
    x.append(MSZoning)
    x.append(LotArea)
    x.append(LotShape)
    x.append(BldgType)
    x.append(OverallCond)
    x.append(Exterior1st)
    x.append(Exterior2nd)
    x.append(YearBuilt)
    x.append(HouseStyle)
    x.append(HeatingQC)

    x.append(FullBath)
    x.append(HalfBath)
    x.append(Bedroom)
    x.append(YrSold)
    x.append(TotRmsAbvGrd)
    x.append(housingPrice)
    dataSorted = pd.DataFrame([x])

    # retrain(dataSorted)

    return templates.TemplateResponse(
        "predict_housing_price_html.html", {"request": request, "housingPrice": ""}
    )


@app.get("/data")
# Route to get all data
# Response: Returns all data (200)
def get_all_data():

    data = getData()
    print(len(json.loads(data)))
    # print(data)
    return {"data": [json.loads(data)], "timestamp": datetime.timestamp(datetime.now())}


# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
