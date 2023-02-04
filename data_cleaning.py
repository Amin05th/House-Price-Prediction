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

df.to_csv("cleaned_train.csv")
