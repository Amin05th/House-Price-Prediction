import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)

df = pd.read_csv("train.csv")

# drop unwanted data
unwanted_columns = ['Id', 'Alley', 'PoolQC', 'MiscFeature', 'Fence', 'MoSold', 'YrSold', 'MSSubClass',
                    'GarageType', 'GarageArea', 'GarageYrBlt', 'GarageFinish', 'YearRemodAdd', 'LandSlope',
                    'BsmtUnfSF', 'BsmtExposure', '2ndFlrSF', 'LowQualFinSF', 'Condition1', 'Condition2', 'Heating',
                    'Exterior1st', 'Exterior2nd', 'HouseStyle', 'LotShape', 'LandContour', 'LotConfig', 'Functional',
                    'BsmtFinSF1', 'BsmtFinSF2', 'FireplaceQu', 'WoodDeckSF', 'GarageQual', 'GarageCond', 'OverallCond'
                    ]

df.drop(columns=unwanted_columns, inplace=True)

# fill missing values
missing_values = ["LotFrontage", "MasVnrType", "MasVnrArea", "BsmtQual", "BsmtCond", "BsmtFinType1", "BsmtFinType2",
                  "Electrical"]

for columns in missing_values:
    df[columns].fillna("No", inplace=True)

# create for non numerical columns dummies

non_numerical_columns = ["MSZoning", "Street", "Utilities", "Neighborhood", "BldgType", "RoofStyle", "RoofMatl",
                         "MasVnrType",
                         "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtFinType1", "BsmtFinType2",
                         "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "PavedDrive", "SaleType",
                         "SaleCondition"]

df = pd.get_dummies(df, columns=non_numerical_columns)

df.to_csv("cleaned_train.csv")