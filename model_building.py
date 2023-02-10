import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LassoLars, SGDRegressor, Ridge, LogisticRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rsquare = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    try:
        rmsle = mean_squared_log_error(y_true, y_pred, squared=False)
    except Exception:
        rmsle = np.nan
    return mae, mse, rsquare, rmse, rmsle


df = pd.read_csv('cleaned_train.csv')
validation_df = pd.read_csv("cleaned_validation.csv")


non_numerical_columns = ["MSZoning", "Street", "Utilities", "Neighborhood", "BldgType", "RoofStyle", "RoofMatl",
                         "MasVnrType",
                         "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtFinType1", "BsmtFinType2",
                         "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "PavedDrive", "SaleType",
                         "SaleCondition"]

encoder = OrdinalEncoder(
    categories=[df["MSZoning"].unique(), df["Street"].unique(), df["Utilities"].unique(), df["Neighborhood"].unique(),
                df["BldgType"].unique(), df["RoofStyle"].unique(), df["RoofMatl"].unique(), df["MasVnrType"].unique(),
                df["ExterQual"].unique(), df["ExterCond"].unique(), df["Foundation"].unique(), df["BsmtQual"].unique(),
                df["BsmtCond"].unique(), df["BsmtFinType1"].unique(), df["BsmtFinType2"].unique(),
                df["HeatingQC"].unique(),
                df["CentralAir"].unique(), df["Electrical"].unique(), df["KitchenQual"].unique(),
                df["PavedDrive"].unique(), df["SaleType"].unique(), df["SaleCondition"].unique()])

df[non_numerical_columns] = encoder.fit_transform(df[non_numerical_columns])
validation_df[non_numerical_columns] = encoder.fit_transform((validation_df[non_numerical_columns]))

# split data in train and test

X = df.drop(columns="SalePrice")
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
param_grid = {"n_estimators": [1400, 1700, 2000]}
model_list = {
    "Linear_Regression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
    "Lasso": LassoLars(alpha=41, eps=1.38, random_state=42),
    "Ridge": Ridge(alpha=1778, random_state=42),
    "SVR": SVR(),
    "Bayes": BayesianRidge(),
    "Gradient_Boosting": GradientBoostingRegressor(random_state=42),
    "Random_Forest": GridSearchCV(RandomForestRegressor(), param_grid=param_grid),
    "SGD_Regressor": SGDRegressor()
}

# preprocessed data with StandardScaler
quantile_escale = QuantileTransformer(n_quantiles=1000)
preprocessed_X_train = quantile_escale.fit_transform(X_train)
preprocessed_X_test = quantile_escale.transform(X_test)

predicted_score = {}
for models in model_list:
    model_list[models].fit(preprocessed_X_train, y_train)
    y_prediction = model_list[models].predict(preprocessed_X_test)
    predicted_score[models] = evaluate(y_test, y_prediction)

score_df = pd.DataFrame(predicted_score, index=["mae", "mse", "rsquare", "rmse", "rmsle"])
score_df.to_csv("score_test_data.csv.csv")


predicted_price = {}
preprocessed_validation_X = quantile_escale.fit_transform(validation_df)
for models in model_list:
    model_list[models].fit(preprocessed_X_train, y_train)
    y_prediction = model_list[models].predict(preprocessed_validation_X)
    predicted_price[models] = y_prediction

predictions = pd.DataFrame(predicted_price)
predictions.to_csv("price_predictions.csv")
