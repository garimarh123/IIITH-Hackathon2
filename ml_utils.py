from os import access
import numpy as np
from pandas.core.frame import DataFrame
from pandas.io.sql import DatabaseError
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from pandas import read_csv
import pandas as pd

# import matplotlib.pyplot as plt
import copy

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import h2o
from h2o.automl import H2OAutoML
from h2o.estimators import H2ORandomForestEstimator
import logging

logging.basicConfig(
    filename="debug.log",
    format="ml_utils:%(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    encoding="utf-8",
    level=logging.DEBUG,
)
page_break = "\n\n========================================================\n"

# define a Gaussain NB classifier
GaussianNBClf = GaussianNB()
GaussianNBClf_acc = 0.0

DecisionTreeClf = DecisionTreeClassifier(max_depth=5, random_state=1)
DecisionTreeClf_acc = 0.0

# define the class encodings and reverse encodings
classes = {0: "Bad Risk", 1: "Good Risk"}
r_classes = {y: x for x, y in classes.items()}

DATA = pd.DataFrame([])
DATA_VIS = pd.DataFrame([])
X_train, X_test, y_train, y_test = (
    pd.DataFrame([]),
    pd.DataFrame([]),
    pd.DataFrame([]),
    pd.DataFrame([]),
)
cat_ix = pd.DataFrame([])

global CLF


def h2o_predict():
    h2o.init()

    credit = h2o.import_file(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    )

    # print(credit)

    # Set the predictors and response;
    # set the response as a factor:
    credit["C21"] = credit["C21"].asfactor()
    predictors = [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
    ]
    response = "C21"

    # Split the dataset into a train and valid set:
    train, valid = credit.split_frame(ratios=[0.8], seed=1234)

    # Build and train the model:
    credit_drf = H2ORandomForestEstimator(
        ntrees=10,
        max_depth=5,
        min_rows=10,
        calibrate_model=True,
        calibration_frame=valid,
        binomial_double_trees=True,
    )
    credit_drf.train(
        x=predictors, y=response, training_frame=train, validation_frame=valid
    )

    # Eval performance:
    perf = credit_drf.model_performance()
    # print(perf)
    # Generate predictions on a validation set (if necessary):
    pred = credit_drf.predict(valid)
    # print(pred)

    exp = credit_drf.explain(valid)


# Function to download the data from the website
def download_data():
    global DATA  # Use x from the global space
    missing_values = ["NA"]
    DATA = read_csv("data/train.csv", sep=",", na_values=missing_values)
    header = "Id,MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle,OverallQual,OverallCond,YearBuilt,YearRemodAdd,RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,MasVnrArea,ExterQual,ExterCond,Foundation,BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,Heating,HeatingQC,CentralAir,Electrical,1stFlrSF,2ndFlrSF,LowQualFinSF,GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,KitchenAbvGr,KitchenQual,TotRmsAbvGrd,Functional,Fireplaces,FireplaceQu,GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,GarageCond,PavedDrive,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,PoolQC,Fence,MiscFeature,MiscVal,MoSold,YrSold,SaleType,SaleCondition,SalePrice".split(
        ","
    )

    print(DATA)
    print(header)
    print(len(header))


# function to prepare the data
def prepare_data():
    global DATA, X_train, X_test, y_train, y_test, cat_ix
    df = copy.deepcopy(DATA)
    logging.debug(df.isnull().sum())
    #     df = df.fillna("")
    # -can be null
    df["LotFrontage"] = df["LotFrontage"].replace(np.nan, 0)
    df["Alley"] = df["Alley"].replace(np.nan, "")
    df["MasVnrType"] = df["MasVnrType"].replace(np.nan, "")
    df["MasVnrArea"] = df["MasVnrArea"].replace(np.nan, "")

    df["BsmtQual"] = df["Alley"].replace(np.nan, "")
    df["BsmtCond"] = df["Alley"].replace(np.nan, "")
    df["BsmtExposure"] = df["BsmtExposure"].replace(np.nan, "")
    df["BsmtFinType1"] = df["BsmtFinType1"].replace(np.nan, "")
    df["BsmtFinType2"] = df["BsmtFinType2"].replace(np.nan, "")
    df["Electrical"] = df["Electrical"].replace(np.nan, "")
    df["FireplaceQu"] = df["FireplaceQu"].replace(np.nan, "")
    df["GarageType"] = df["GarageType"].replace(np.nan, "")
    df["GarageYrBlt"] = df["GarageYrBlt"].replace(np.nan, 0)
    df["GarageFinish"] = df["GarageFinish"].replace(np.nan, "")
    df["GarageCond"] = df["GarageCond"].replace(np.nan, "")
    df["GarageQual"] = df["GarageQual"].replace(np.nan, "")

    df["PoolQC"] = df["PoolQC"].replace(np.nan, "")
    df["Fence"] = df["Fence"].replace(np.nan, "")
    df["MiscFeature"] = df["MiscFeature"].replace(np.nan, "")

    print("null value: " + str(df.isnull().sum().sum()))
    print(df.isnull().sum().sort_values(ascending=True))
    y = df.SalePrice
    X = df.iloc[:, :-1]

    # Categorical features has to be converted into integer values for the model to process.
    # This is done through one hot encoding.
    # select categorical features
    cat_ix = X.select_dtypes(include=["string", "bool", "int", "float"]).columns

    # one hot encode categorical features only
    ct = ColumnTransformer([("o", OneHotEncoder(), cat_ix)], remainder="passthrough")
    X = ct.fit_transform(X)
    # label encode the target variable to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)
    #     print(X.shape, y.shape)
    #     print(type(X), type(y))
    #     print(X)
    # Splitting the data for training and testing
    return X, y


# function to train and load the model during startup
def load_model():
    # load the dataset
    global CLF
    download_data()
    X_train, y_train = prepare_data()

    # define a Gaussain NB classifier
    GaussianNBClf = GaussianNB()
    DecisionTreeClf = DecisionTreeClassifier(max_depth=5, random_state=1)
    RandomForestClf = RandomForestClassifier(
        criterion="entropy", random_state=0, min_samples_split=2
    )

    GaussianNBClf.fit(X_train.toarray(), y_train)
    # calculate the print the accuracy score


#     GaussianNBClf_acc = accuracy_score(y_test, GaussianNBClf.predict(X_test))

#     DecisionTreeClf.fit(X_train, y_train)
#     DecisionTreeClf_acc = accuracy_score(y_test, DecisionTreeClf.predict(X_test))

#     RandomForestClf.fit(X_train, y_train)
#     RandomForestClf_acc = accuracy_score(y_test, RandomForestClf.predict(X_test))

#     if (
#         DecisionTreeClf_acc > GaussianNBClf_acc
#         and DecisionTreeClf_acc > RandomForestClf_acc
#     ):
#         print()
#         CLF = DecisionTreeClf
#         return DecisionTreeClf_acc, "Decision Tree Classifier"
#     else:
#         if (
#             GaussianNBClf_acc > DecisionTreeClf_acc
#             and GaussianNBClf_acc > RandomForestClf_acc
#         ):
#             CLF = GaussianNBClf
#             return GaussianNBClf_acc, "GaussianNB Classifier"
#         else:
#             CLF = RandomForestClf
#             return RandomForestClf_acc, "RandomForest Classifier"


# function to predict the flower using the model
def predict(query_data):
    global DATA, cat_ix

    # new_data = list(query_data.dict().values())
    new_data = query_data
    new_data.append(1)
    new_data = pd.DataFrame([new_data])

    x = copy.deepcopy(DATA)
    x = x.append(new_data, ignore_index=True)
    last_ix = len(x.columns) - 1
    x = x.drop(last_ix, axis=1)

    ct = ColumnTransformer([("o", OneHotEncoder(), cat_ix)], remainder="passthrough")
    x = ct.fit_transform(x.astype(object))

    prediction = CLF.predict([x[x.shape[0] - 1]])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]


# function to retrain the model as part of the feedback loop
def retrain(data):

    global DATA, X_train, y_train, X_test, y_test

    DATA = DATA.append(pd.DataFrame(data), ignore_index=True)

    prepare_data()

    CLF.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, CLF.predict(X_test))


def substitute(df):
    headers = [
        "Status of existing checking account",
        "Duration in month",
        "Credit history",
        "Purpose",
        "Credit amount",
        "Savings account/bonds",
        "Present employment since",
        "Installment rate in percentage of disposable income",
        "Personal status and sex",
        "Other debtors / guarantors",
        "Present residence since",
        "Property",
        "Age in years",
        "Other installment plans",
        "Housing",
        "Number of existing credits at this bank",
        "Job",
        "Number of people being liable to provide maintenance for",
        "Telephone",
        "foreign worker",
        "Risk",
    ]
    df.columns = headers
    # df.to_csv("german_data_credit_cat.csv",index=False) #save as csv file

    # for structuring only
    Status_of_existing_checking_account = {
        "A14": "no checking account",
        "A11": "<0 DM",
        "A12": "0 <= <200 DM",
        "A13": ">= 200 DM ",
    }
    df["Status of existing checking account"] = df[
        "Status of existing checking account"
    ].map(Status_of_existing_checking_account)

    Credit_history = {
        "A34": "critical account",
        "A33": "delay in paying off",
        "A32": "existing credits paid back duly till now",
        "A31": "all credits at this bank paid back duly",
        "A30": "no credits taken",
    }
    df["Credit history"] = df["Credit history"].map(Credit_history)

    Purpose = {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
    }
    df["Purpose"] = df["Purpose"].map(Purpose)

    Saving_account = {
        "A65": "no savings account",
        "A61": "<100 DM",
        "A62": "100 <= <500 DM",
        "A63": "500 <= < 1000 DM",
        "A64": ">= 1000 DM",
    }
    df["Savings account/bonds"] = df["Savings account/bonds"].map(Saving_account)

    Present_employment = {
        "A75": ">=7 years",
        "A74": "4<= <7 years",
        "A73": "1<= < 4 years",
        "A72": "<1 years",
        "A71": "unemployed",
    }
    df["Present employment since"] = df["Present employment since"].map(
        Present_employment
    )

    Personal_status_and_sex = {
        "A95": "female:single",
        "A94": "male:married/widowed",
        "A93": "male:single",
        "A92": "female:divorced/separated/married",
        "A91": "male:divorced/separated",
    }
    df["Personal status and sex"] = df["Personal status and sex"].map(
        Personal_status_and_sex
    )

    Other_debtors_guarantors = {
        "A101": "none",
        "A102": "co-applicant",
        "A103": "guarantor",
    }
    df["Other debtors / guarantors"] = df["Other debtors / guarantors"].map(
        Other_debtors_guarantors
    )

    Property = {
        "A121": "real estate",
        "A122": "savings agreement/life insurance",
        "A123": "car or other",
        "A124": "unknown / no property",
    }
    df["Property"] = df["Property"].map(Property)

    Other_installment_plans = {"A143": "none", "A142": "store", "A141": "bank"}
    df["Other installment plans"] = df["Other installment plans"].map(
        Other_installment_plans
    )

    Housing = {"A153": "for free", "A152": "own", "A151": "rent"}
    df["Housing"] = df["Housing"].map(Housing)

    Job = {
        "A174": "management/ highly qualified employee",
        "A173": "skilled employee / official",
        "A172": "unskilled - resident",
        "A171": "unemployed/ unskilled  - non-resident",
    }
    df["Job"] = df["Job"].map(Job)

    Telephone = {"A192": "yes", "A191": "none"}
    df["Telephone"] = df["Telephone"].map(Telephone)

    foreign_worker = {"A201": "yes", "A202": "no"}
    df["foreign worker"] = df["foreign worker"].map(foreign_worker)

    risk = {1: "Good Risk", 2: "Bad Risk"}
    df["Risk"] = df["Risk"].map(risk)
    return df


def formatData(df):
    last_ix = len(df.columns) - 1
    X, y = df.drop(last_ix, axis=1), df[last_ix]
    cat_ix = X.select_dtypes(include=["object", "bool"]).columns
    # one hot encode categorical features only
    ct = ColumnTransformer([("o", OneHotEncoder(), cat_ix)], remainder="passthrough")
    X = ct.fit_transform(X)
    # label encode the target variable to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)
    return X, y


def getData():
    global DATA
    print("test")
    data = substitute(copy.deepcopy(DATA))
    data_arr = data.to_numpy()
    print(data_arr)
    js = data.to_json(orient="records")
    print(data.columns)
    text = "\s"
    js.replace(text[0], "")

    return js


def explainability():
    h2o.init()

    # Reponse column
    y = "Risk"

    # Run AutoML for 1 minute
    aml = H2OAutoML(max_runtime_secs=60, seed=1)
    train = pd.concat(
        [pd.DataFrame(X_train), pd.DataFrame(y_train, columns=["Risk"])], axis=1
    )
    print(train)
    test = pd.concat(
        [pd.DataFrame(X_test), pd.DataFrame(y_test, columns=["Risk"])], axis=1
    )
    aml.train(y=y, training_frame=h2o.H2OFrame(train))

    # Explain leader model & compare with all AutoML models
    exa = aml.explain(h2o.H2OFrame(test))

    # Explain a single H2O model (e.g. leader model from AutoML)
    exm = aml.leader.explain(h2o.H2OFrame(test))

    # h2o.init()
    # X, y = formatData(DATA)
    # print(y)
    # # Run AutoML for 1 minute
    # aml = H2OAutoML(max_runtime_secs=60, seed=1)
    # train = X
    # print(train)
    # # Reponse column
    # y = "Risk"

    # aml.train(y=y, training_frame=h2o.H2OFrame(train))

    # # Explain leader model & compare with all AutoML models
    # exa = aml.explain(X_test)

    # # Explain a single H2O model (e.g. leader model from AutoML)
    # exm = aml.leader.explain(h2o.H2OFrame(X_test))
    return


load_model()
# load_model()
# explainability()
# # predict([["A11", "6", "A34", "A43", "1169", "A65", "A75", "4", "A93", "A101", "4", "A121", "67",	"A143", "A152", "2", "A173",	"1",	"A192", "A201"]])
# predict([0.000e+00, 1.000e+00, 0.000e+00 ,0.000e+00 ,0.000e+00, 0.000e+00 ,0.000e+00,
#  0.000e+00 ,1.000e+00, 0.000e+00 ,0.000e+00 ,0.000e+00 ,0.000e+00, 0.000e+00,
#  0.000e+00, 0.000e+00 ,0.000e+00 ,0.000e+00 ,1.000e+00 ,1.000e+00, 0.000e+00,
#  0.000e+00, 0.000e+00 ,0.000e+00, 0.000e+00 ,0.000e+00 ,0.000e+00, 1.000e+00,
#  0.000e+00 ,0.000e+00 ,0.000e+00, 1.000e+00 ,0.000e+00 ,1.000e+00, 0.000e+00,
#  0.000e+00, 0.000e+00 ,1.000e+00, 0.000e+00 ,0.000e+00 ,0.000e+00, 0.000e+00,
#  1.000e+00, 0.000e+00 ,1.000e+00 ,0.000e+00, 0.000e+00 ,0.000e+00, 1.000e+00,
#  0.000e+00 ,1.000e+00 ,0.000e+00 ,1.000e+00, 0.000e+00 ,2.100e+01 ,3.652e+03,
#  2.000e+00 ,3.000e+00, 2.700e+01, 2.000e+00 ,1.000e+00])
# print("Model actual: Bad Risk")

# print("=======================")
# print("Retrain model")
# df=read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",\
#                sep=" ",header=None)

# retrain(df)
# predict([0.000e+00, 1.000e+00, 0.000e+00 ,0.000e+00 ,0.000e+00, 0.000e+00 ,0.000e+00,
#  0.000e+00 ,1.000e+00, 0.000e+00 ,0.000e+00 ,0.000e+00 ,0.000e+00, 0.000e+00,
#  0.000e+00, 0.000e+00 ,0.000e+00 ,0.000e+00 ,1.000e+00 ,1.000e+00, 0.000e+00,
#  0.000e+00, 0.000e+00 ,0.000e+00, 0.000e+00 ,0.000e+00 ,0.000e+00, 1.000e+00,
#  0.000e+00 ,0.000e+00 ,0.000e+00, 1.000e+00 ,0.000e+00 ,1.000e+00, 0.000e+00,
#  0.000e+00, 0.000e+00 ,1.000e+00, 0.000e+00 ,0.000e+00 ,0.000e+00, 0.000e+00,
#  1.000e+00, 0.000e+00 ,1.000e+00 ,0.000e+00, 0.000e+00 ,0.000e+00, 1.000e+00,
#  0.000e+00 ,1.000e+00 ,0.000e+00 ,1.000e+00, 0.000e+00 ,2.100e+01 ,3.652e+03,
#  2.000e+00 ,3.000e+00, 2.700e+01, 2.000e+00 ,1.000e+00])
# print("Model actual: Bad Risk")
