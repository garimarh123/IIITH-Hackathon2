# IIITH Hackathon 2

# Overview of data

**The information about the dataset is in the readme file, in the 'data' folder**

# Explainability of Data

<!-- TODO: Remove before submission -->

Link: https://colab.research.google.com/drive/1nPJ7uy2hHJpA5efjSGt_7ORVPWWVMV7A#scrollTo=x_Fn6k3_hBti

## Variable Importance

The variable importance plot shows the relative importance of the most important variables in the model.
![image](https://user-images.githubusercontent.com/55736158/132108557-f5877aa5-73bf-4c02-8990-3f467f784034.png)

## Variable Importance Heatmap

Variable importance heatmap shows variable importance across multiple models. Some models in H2O return variable importance for one-hot (binary indicator) encoded versions of categorical columns (e.g. Deep Learning, XGBoost). In order for the variable importance of categorical columns to be compared across all model types we compute a summarization of the the variable importance across all one-hot encoded features and return a single variable importance for the original categorical feature. By default, the models and variables are ordered by their similarity.
![image](https://user-images.githubusercontent.com/55736158/132108592-24283666-503e-4bef-9614-1fc7b881fae5.png)

## SHAP Summary

SHAP summary plot shows the contribution of the features for each instance (row of data). The sum of the feature contributions and the bias term is equal to the raw prediction of the model, i.e., prediction before applying inverse link function.
![image](https://user-images.githubusercontent.com/55736158/132108613-5e46468a-1769-42c5-bc0c-daf502e99a59.png)

## What variables are more important

By looking at the Variable Importance, Variable Important Heatmap and the SHAP summary diagrams, it is very clear that the **OverallQual(Overall material and finish quality)** is the most important variable, followed by:

- ExterQual (Exterior material quality)
- GrLivArea (Above grade/groundliving area square feet)
- KitchenQual (Kitchen quality)
- Neighborhood (Physical locations within Ames city limits)
- 1stFlrSF (First Floor square feet)
- TotalBsmtSF (Total square feet of basement area)
- BsmtFinSF1 (Type 1 finished square feet)
- GarageCars (Size of garage in car capacity)
- GarageArea (Size of garage in square feet)

  It is very apparent that the buyers give a lot of importance to the finishing quality and kitchen quality of the house, the size of the garages, basements and first floor size as well the neighborhood and the area.

These are some of the attributes that we have identified, based on some research, that also contribute heavily to a buyer's decision as they are all basic things a buyer will consider when looking for a new house:

- MSSubClass
- MSZoning
- LotArea
- LotShape
- BldgType
- OverallCond
- Exterior1st
- Exterior2nd
- YearBuilt
- HouseStyle
- HeatingQC
- FullBath
- HalfBath
- Bedroom
- YrSold
- TotRmsAbvGrd

  **Therefore we will be using only the top 10 attributes and 18 others listed to train our model and not all 79 attributes**

# Approach to problem

A regression analysis is used to model the relationship between a dependent variable and one or more independent variables.
It is quite evident that this is a regression problem as the selling price is dependent on the other 79 attributes (x attributes)

The overall idea of regression is to examine two things:

- Does a set of predictor attributes do a good job in predicting an outcome (selling price)?
- Which attributes in particular are significant predictors of the outcome, and in what way do they indicated by the magnitude and sign of the beta estimates–impact the outcome?

# NOTES FOR US => will be removed before submission

## How to choose the correct regression model?

**Link:** https://www.listendata.com/2018/03/regression-analysis.html

1. If dependent variable is continuous and model is suffering from collinearity or there are a lot of independent variables, you can try PCR, PLS, ridge, lasso and elastic net regressions. You can select the final model based on Adjusted r-square, RMSE, AIC and BIC.
2. If you are working on count data, you should try poisson, quasi-poisson and negative binomial regression.
3. To avoid overfitting, we can use cross-validation method to evaluate models used for prediction. We can also use ridge, lasso and elastic net regressions techniques to correct overfitting issue.
4. Try support vector regression when you have non-linear model.

**Linear Regression:** It is the simplest form of regression. It is a technique in which the dependent variable is continuous in nature. The relationship between the dependent variable and independent variables is assumed to be linear in nature.

**Quantile regression:** Quantile regression is the extension of linear regression and we generally use it when outliers, high skeweness and heteroscedasticity exist in the data.

**Principal Components Regression (PCR):** PCR is a regression technique which is widely used when you have many independent variables OR multicollinearity exist in your data. It is divided into 2 steps:

1. Getting the Principal components
2. Run regression analysis on principal components

The most common features of PCR are:

1. Dimensionality Reduction
2. Removal of multicollinearity
