# House-Price-Prediction: Project Overview

- Created a model that calculates House Prices for people who are looking for houses with good price ratio
- Downloaded Kaggle dataset with over than 3000 Houses
- Optimized Linear, Lasso, Ridge, DecisionTreeRegressor, Random Forest Regressor using GridsearchCV, SVR, Bayes, Gradient_Boosting and SGDRegressor to reach best model
- Built a client facing User Interface using streamlit

## Code and Resources Used

**Python Version:** 3.10
**Packages:** numpy, pandas, sklearn, matplotlib, seaborn and streamlit
**Kaggle Dataset:** [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv)

## Downloaded Dataset

- SalePrice
- MSSubClass
- MSZoning
- LotFrontage
- LotArea
- Street
- Alley
- LotShape
- LandContour
- Utilities
- LotConfig
- LandSlope
- Neighborhood
- Condition1
- Condition2
- BldgType
- HouseStyle
- OverallQual
- OverallCond
- YearBuilt
- YearRemodAdd
- RoofStyle
- RoofMatl
- Exterior1st
- Exterior2nd
- MasVnrType
- MasVnrArea
- ExterQual
- ExterCond
- Foundation
- BsmtQual
- BsmtCond
- BsmtExposure
- BsmtFinType1
- BsmtFinSF1
- BsmtFinType2
- BsmtFinSF2
- BsmtUnfSF
- TotalBsmtSF
- Heating
- HeatingQC
- CentralAir
- Electrical
- 1stFlrSF
- 2ndFlrSF
- LowQualFinSF
- GrLivArea
- BsmtFullBath
- BsmtHalfBath
- FullBath
- HalfBath
- Bedroom
- Kitchen
- KitchenQual
- TotRmsAbvGrd
- Functional
- Fireplaces
- FireplaceQu
- GarageType
- GarageYrBlt
- GarageFinish
- GarageCars
- GarageArea
- GarageQual
- GarageCond
- PavedDrive
- WoodDeckSF
- OpenPorchSF
- EnclosedPorch
- 3SsnPorch
- ScreenPorch
- PoolArea
- PoolQC
- Fence
- MiscFeature
- MiscVal
- MoSold
- YrSold
- SaleType
- SaleCondition


## Data Cleaning

After downloading the data, I have to clean it up for a better prediction and that the data is usable for our model. I have made the following changes

- dropped unwanted columns
- replaced missing numerical values with -1
- dropped rows with object missing values


## EDA

- created heatmap using seaborn
- created pairplot using seaborn
- created violin plot using matplotlib

## Model Building

First I transformed columns using OrdinalEncoder then I scaled them using StandardScaler and at the end I split the data into train and tests sets with a test size of 20%

I tried 9 different models:

- **Linear Regresson:** Baseline for the model
- **DecisionTreeregressor:** To see how one Tree is doing
- **Lasso:** To test Lasso out
- **Ridge:** To test Ridge out
- **SVR:** To test SVR out
- **Bayes:** To test Bayes out
- **Gradient Boosting:** To try Gradient Boosting out
- **Random Forest:** I thought a Random Forest Regressor with Parameter tuning would be a good fit
- ** SGDRegressor:** To test SGDRegressor out

## Productionization

In this step I built a streamlit App that shows the diffrent Attributes of a house and the price




