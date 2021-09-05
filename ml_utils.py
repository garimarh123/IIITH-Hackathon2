import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

    GaussianNBClf = GaussianNB()
    DecisionTreeClf = DecisionTreeClassifier(max_depth=2, random_state=1)
    RandomForestClf = RandomForestClassifier(
        criterion="entropy", random_state=0, min_samples_split=2
    )

    GaussianNBClf.fit(X_train, y_train)
    # calculate the print the accuracy score
    GaussianNBClf_acc = accuracy_score(y_test, GaussianNBClf.predict(X_test))

    DecisionTreeClf.fit(X_train, y_train)
    DecisionTreeClf_acc = accuracy_score(y_test, DecisionTreeClf.predict(X_test))

    RandomForestClf.fit(X_train, y_train)
    RandomForestClf_acc = accuracy_score(y_test, RandomForestClf.predict(X_test))
    print(f"GaussianNB Classifier trained with accuracy: " + str(GaussianNBClf_acc))
    print(
        f"Decision Tree Classifier trained with accuracy: " + str(DecisionTreeClf_acc)
    )
    print(
        f"Random Forest Classifier trained with accuracy: " + str(RandomForestClf_acc)
    )
    if (
        DecisionTreeClf_acc > GaussianNBClf_acc
        and DecisionTreeClf_acc > RandomForestClf_acc
    ):
        print("Using Decision Tree Classifier")
        print()
        CLF = DecisionTreeClf
        print(page_break)
        return DecisionTreeClf_acc, "Decision Tree Classifier"
    else:
        if (
            GaussianNBClf_acc > DecisionTreeClf_acc
            and GaussianNBClf_acc > RandomForestClf_acc
        ):
            print("Using GaussianNB Classifier")
            CLF = GaussianNBClf
            print(page_break)
            return GaussianNBClf_acc, "GaussianNB Classifier"
        else:
            print("Using RandomForest Classifier")
            CLF = RandomForestClf
            print(page_break)
            return RandomForestClf_acc, "RandomForest Classifier"


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
