import numpy as np
from sklearn.naive_bayes import GaussianNB
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
    # "LotArea",
    # "LotShape",
    # "BldgType",
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

DATA = pd.DataFrame([])
cat_ix = pd.DataFrame([])

global CLF

# function to prepare the data
def prepare_data():
    global DATA, X_train, X_test, y_train, y_test, cat_ix
    df = copy.deepcopy(DATA)

    print(page_break, "Columns with null values")
    missing_val_count_by_column = df.isnull().sum()
    print(missing_val_count_by_column[missing_val_count_by_column > 0])

    # print(df.isnull().sum())
    #     df = df.fillna("")
    # -can be null
    # df["LotFrontage"] = df["LotFrontage"].replace(np.nan, 0)
    # df["Alley"] = df["Alley"].replace(np.nan, "")
    # df["MasVnrType"] = df["MasVnrType"].replace(np.nan, "")
    # df["MasVnrArea"] = df["MasVnrArea"].replace(np.nan, "")

    # df["BsmtQual"] = df["Alley"].replace(np.nan, "")
    # df["BsmtCond"] = df["Alley"].replace(np.nan, "")
    # df["BsmtExposure"] = df["BsmtExposure"].replace(np.nan, "")
    # df["BsmtFinType1"] = df["BsmtFinType1"].replace(np.nan, "")
    # df["BsmtFinType2"] = df["BsmtFinType2"].replace(np.nan, "")
    # df["Electrical"] = df["Electrical"].replace(np.nan, "")
    # df["FireplaceQu"] = df["FireplaceQu"].replace(np.nan, "")
    # df["GarageType"] = df["GarageType"].replace(np.nan, "")
    # df["GarageYrBlt"] = df["GarageYrBlt"].replace(np.nan, 0)
    # df["GarageFinish"] = df["GarageFinish"].replace(np.nan, "")
    # df["GarageCond"] = df["GarageCond"].replace(np.nan, "")
    # df["GarageQual"] = df["GarageQual"].replace(np.nan, "")

    # df["PoolQC"] = df["PoolQC"].replace(np.nan, "")
    # df["Fence"] = df["Fence"].replace(np.nan, "")
    # df["MiscFeature"] = df["MiscFeature"].replace(np.nan, "")

    y = df.SalePrice
    X = df.iloc[:, :-1]

    # print("null value: " + str(X.isnull().sum().sum()))
    # print(df.isnull().sum().sort_values(ascending=True))
    # # Categorical features has to be converted into integer values for the model to process.
    # # This is done through one hot encoding.
    # # select categorical features
    # cat_ix = X.select_dtypes(include=["string", "bool", "int", "float"]).columns

    # # one hot encode categorical features only
    # ct = ColumnTransformer([("o", OneHotEncoder(), cat_ix)], remainder="passthrough")
    # X = ct.fit_transform(X)

    #     print(X.shape, y.shape)
    #     print(type(X), type(y))
    #     print(X)
    # Splitting the data for training and testing
    return X, y


# function to train and load the model during startup
def load_model():
    # load the dataset
    global CLF
    global DATA  # Use x from the global space
    missing_values = ["NA"]
    DATA = read_csv("data/train.csv", sep=",", na_values=missing_values)
    # header = "Id,MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle,OverallQual,OverallCond,YearBuilt,YearRemodAdd,RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,MasVnrArea,ExterQual,ExterCond,Foundation,BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,Heating,HeatingQC,CentralAir,Electrical,1stFlrSF,2ndFlrSF,LowQualFinSF,GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,KitchenAbvGr,KitchenQual,TotRmsAbvGrd,Functional,Fireplaces,FireplaceQu,GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,GarageCond,PavedDrive,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,PoolQC,Fence,MiscFeature,MiscVal,MoSold,YrSold,SaleType,SaleCondition,SalePrice".split(
    #     ","
    # )
    print(attributes_for_training_model)
    print(len(attributes_for_training_model))
    X_train, y_train = prepare_data()

    # define a Gaussain NB classifier
    CLF = GaussianNB()

    # CLF.fit(X_train.toarray(), y_train)
    # calculate the print the accuracy score
    # GaussianNBClf_acc = accuracy_score(y_test, CLF.predict(X_test))


# function to predict the selling price using the model
def predict(query_data):
    global DATA, cat_ix

    new_data = query_data
    new_data.append(1)
    new_data = pd.DataFrame([new_data])

    # This was nessassry to encode the data properly in the previous hackathon,
    # I will remove it if it's needed, when we get here
    x = copy.deepcopy(DATA)
    x = x.append(new_data, ignore_index=True)
    last_ix = len(x.columns) - 1
    x = x.drop(last_ix, axis=1)

    ct = ColumnTransformer([("o", OneHotEncoder(), cat_ix)], remainder="passthrough")
    x = ct.fit_transform(x.astype(object))

    prediction = CLF.predict([x[x.shape[0] - 1]])[0]
    print(f"Model prediction: {prediction}")
    return prediction


load_model()
