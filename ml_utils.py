import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score

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

trainData = pd.DataFrame([])
cat_ix = pd.DataFrame([])

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
    global CLF, trainData, cat_ix

    missing_values = ["NA"]
    trainData = read_csv("data/train.csv", sep=",", na_values=missing_values)
    trainData.columns = "Id,MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle,OverallQual,OverallCond,YearBuilt,YearRemodAdd,RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,MasVnrArea,ExterQual,ExterCond,Foundation,BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,Heating,HeatingQC,CentralAir,Electrical,1stFlrSF,2ndFlrSF,LowQualFinSF,GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,Bedroom,KitchenAbvGr,KitchenQual,TotRmsAbvGrd,Functional,Fireplaces,FireplaceQu,GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,GarageCond,PavedDrive,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,PoolQC,Fence,MiscFeature,MiscVal,MoSold,YrSold,SaleType,SaleCondition,SalePrice".split(
        ","
    )
    testData = read_csv("data/train.csv", sep=",", na_values=missing_values)
    testData.columns = "Id,MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle,OverallQual,OverallCond,YearBuilt,YearRemodAdd,RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,MasVnrArea,ExterQual,ExterCond,Foundation,BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,Heating,HeatingQC,CentralAir,Electrical,1stFlrSF,2ndFlrSF,LowQualFinSF,GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,Bedroom,KitchenAbvGr,KitchenQual,TotRmsAbvGrd,Functional,Fireplaces,FireplaceQu,GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,GarageCond,PavedDrive,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,PoolQC,Fence,MiscFeature,MiscVal,MoSold,YrSold,SaleType,SaleCondition,SalePrice".split(
        ","
    )
    print("Loaded data")
    workingTrainingSet = copy.deepcopy(trainData)
    workingTrainingSet.append(testData)

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

    X_test, y_test = prepare_data(testData)
    print("Data prepared")

    LinearRegressionModel = LinearRegression()

    RandomForestModel = RandomForestRegressor(n_estimators=1000, random_state=42)

    LinearRegressionModel.fit(X_train, y_train)
    LinearRegressionModelPredictions = LinearRegressionModel.predict(X_test)
    LinearRegressionModelErrors = abs(LinearRegressionModelPredictions - y_test)
    mape = 100 * (LinearRegressionModelErrors / y_test)
    LinearRegressionModel_acc = 100 - np.mean(mape)

    RandomForestModel.fit(X_train, y_train)
    RandomForestModelPredictions = RandomForestModel.predict(X_test)
    RandomForestModelErrors = abs(RandomForestModelPredictions - y_test)
    mape = 100 * (RandomForestModelErrors / y_test)
    RandomForestModel_acc = 100 - np.mean(mape)

    print(
        f"Linear Regression Model trained with accuracy: "
        + str(LinearRegressionModel_acc)
    )
    print(f"Random Forest Model trained with accuracy: " + str(RandomForestModel_acc))

    if LinearRegressionModel_acc > RandomForestModel_acc:
        print("Using Linear Regression Model ")
        CLF = LinearRegressionModel
        print(page_break)
        return LinearRegressionModel_acc, "Linear Regression Model "
    else:
        print("Using Random Forest Model")
        CLF = RandomForestModel
        print(page_break)
        return RandomForestModel_acc, "Random Forest Model"


# function to predict the selling price using the model
def predict(query_data):
    global DATA, cat_ix

    new_data = query_data
    new_data.append(1)
    new_data = pd.DataFrame([new_data])[attributes_for_training_model]

    # This was nessassry to encode the data properly in the previous hackathon,
    # I will remove it if it's needed, when we get here
    x = copy.deepcopy(trainData)
    x = x.append(new_data, ignore_index=True)
    last_ix = len(x.columns) - 1
    x = x.drop(last_ix, axis=1)

    ct = ColumnTransformer([("o", OneHotEncoder(), cat_ix)], remainder="passthrough")
    x = ct.fit_transform(x.astype(object))

    prediction = CLF.predict([x[x.shape[0] - 1]])[0]
    print(f"Model prediction: {prediction}")
    return prediction


load_model()
