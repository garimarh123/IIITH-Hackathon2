import numpy as np
from numpy.lib import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from pandas import read_csv
import pandas as pd

# import matplotlib.pyplot as plt
import copy

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# if we printing a lot of things we can used this as a divider
page_break = "\n\n========================================================\n"
attributes_for_training_model = [
    "ExterQual",
    "GrLivArea",
    "KitchenQual",
    "Neighborhood",
    "1stFlrSF",
    "TotalBsmtSF",
    "BsmtFinSF1",
    "GarageCars",
    "GarageArea",
    "MSSubClass",
    "MSZoning",
    "LotArea",
    "LotShape",
    "BldgType",
    "OverallCond",
    "Exterior1st",
    "Exterior2nd",
    "YearBuilt",
    "HouseStyle",
    "HeatingQC",
    "FullBath",
    "HalfBath",
    "Bedroom",
    "YrSold",
    "TotRmsAbvGrd",
]
all_attributes_for_training_model = [
    "ExterQual",
    "GrLivArea",
    "KitchenQual",
    "Neighborhood",
    "1stFlrSF",
    "TotalBsmtSF",
    "BsmtFinSF1",
    "GarageCars",
    "GarageArea",
    "MSSubClass",
    "MSZoning",
    "LotArea",
    "LotShape",
    "BldgType",
    "OverallCond",
    "Exterior1st",
    "Exterior2nd",
    "YearBuilt",
    "HouseStyle",
    "HeatingQC",
    "FullBath",
    "HalfBath",
    "Bedroom",
    "YrSold",
    "TotRmsAbvGrd",
    "SalePrice",
]
trainData = pd.DataFrame([])
cat_ix = pd.DataFrame([])

missing_values = ["NA"]

global CLF


def checkIfNullsAndMissingVariables(df):
    print(page_break, "Columns with null values")
    missing_val_count_by_column = df.isnull().sum()
    missing_val_count_by_column = missing_val_count_by_column[
        missing_val_count_by_column > 0
    ].to_frame()
    isAllAtrributesUsedValid = missing_val_count_by_column.isin(
        attributes_for_training_model
    ).any()
    print("Missing or nulls : " + isAllAtrributesUsedValid)
    return isAllAtrributesUsedValid


# function to prepare the data
def prepare_data(df):
    global cat_ix

    # checkIfNullsAndMissingVariables(df)
    y = df.SalePrice
    X = df[attributes_for_training_model]

    # one hot encode categorical features only
    ct = ColumnTransformer([("o", OneHotEncoder(), cat_ix)], remainder="passthrough")
    X = ct.fit_transform(X)

    return X.toarray(), y


# function to train and load the model during startup
def load_model():
    # load the dataset
    global CLF, trainData, cat_ix, X_test, y_test

    trainData = read_csv("data/train.csv", sep=",", na_values=missing_values)
    trainData = substitute(copy.deepcopy(trainData))
    
    print("Loaded data")

    workingTrainingSet = trainData

    # Categorical features has to be converted into integer values for the model to process.
    # This is done through one hot encoding.
    # select categorical features

    cat_ix = (
        workingTrainingSet[attributes_for_training_model]
        .iloc[:, :-1]
        .select_dtypes(include=["object", "bool"])
        .columns
    )

    X_train, y_train = prepare_data(trainData)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.3)
    print("Data prepared")

    RandomForestModel = RandomForestRegressor(n_estimators=1000, random_state=42)

    RandomForestModel.fit(X_train, y_train)
    RandomForestModelPredictions = RandomForestModel.predict(X_test)
    RandomForestModelErrors = abs(RandomForestModelPredictions - y_test)
    mape = 100 * (RandomForestModelErrors / y_test)
    RandomForestModel_acc = 100 - np.mean(mape)

    print(f"Random Forest Model trained with accuracy: " + str(RandomForestModel_acc))

    CLF = RandomForestModel
    print(page_break)
    return RandomForestModel_acc, "Random Forest Model"


# function to predict the selling price using the model
def predict_with_test_set():
    global cat_ix
    testData = read_csv("data/test.csv", sep=",", na_values=missing_values)
    testData = substitute(copy.deepcopy(testData))

    X_test, y_test = prepare_data(testData)

    RandomForestModelPredictions = CLF.predict(X_test)
    RandomForestModelErrors = abs(RandomForestModelPredictions - y_test)
    mape = 100 * (RandomForestModelErrors / y_test)
    RandomForestModel_acc = 100 - np.mean(mape)

    print(f"Random Forest Model accuracy with the Test dataset: " + str(RandomForestModel_acc))
    print(page_break)

    return RandomForestModel_acc


def substitute(df):
    headers = "Id,MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle,OverallQual,OverallCond,YearBuilt,YearRemodAdd,RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,MasVnrArea,ExterQual,ExterCond,Foundation,BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,Heating,HeatingQC,CentralAir,Electrical,1stFlrSF,2ndFlrSF,LowQualFinSF,GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,Bedroom,KitchenAbvGr,KitchenQual,TotRmsAbvGrd,Functional,Fireplaces,FireplaceQu,GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,GarageCond,PavedDrive,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,PoolQC,Fence,MiscFeature,MiscVal,MoSold,YrSold,SaleType,SaleCondition,SalePrice".split(
        ","
    )
    df.columns = headers
    return df

