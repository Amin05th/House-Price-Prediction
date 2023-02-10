import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)

df = pd.read_csv("train.csv")
validation_df = pd.read_csv("validation.csv")

# drop unwanted data
unwanted_columns = ['Id', 'Alley', 'PoolQC', 'MiscFeature', 'Fence', 'MoSold', 'YrSold', 'MSSubClass',
                    'GarageType', 'GarageArea', 'GarageYrBlt', 'GarageFinish', 'YearRemodAdd', 'LandSlope',
                    'BsmtUnfSF', 'BsmtExposure', '2ndFlrSF', 'LowQualFinSF', 'Condition1', 'Condition2', 'Heating',
                    'Exterior1st', 'Exterior2nd', 'HouseStyle', 'LotShape', 'LandContour', 'LotConfig', 'Functional',
                    'BsmtFinSF1', 'BsmtFinSF2', 'FireplaceQu', 'WoodDeckSF', 'GarageQual', 'GarageCond', 'OverallCond'
                    ]

df.drop(columns=unwanted_columns, inplace=True)
validation_df.drop(columns=unwanted_columns, inplace=True)

# fill missing values
missing_numerical_values = ["LotFrontage", "MasVnrType", "MasVnrArea", "BsmtQual", "BsmtCond", "BsmtFinType1",
                            "BsmtFinType2",
                            "Electrical", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "GarageCars"]

for columns in missing_numerical_values:
    df[columns].fillna(-1, inplace=True)
    validation_df[columns].fillna(-1, inplace=True)

validation_df.dropna(how="any", axis=0, inplace=True)

df.to_csv("cleaned_train.csv")
validation_df.to_csv("cleaned_validation.csv")
