# IIITH Hackathon 2

# Explainability of Data
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
By looking at the Variable Importance, Variable Important Heatmap and the SHAP summary diagrams, it is very clear that 
