import numpy as np
from sklearn import linear_model
from sklearn.metrics import explained_variance_score
from house_eda_jup import price_data_preparation, transform_house_pd
from regression_plots import create_partial_plots
import pandas as pd


def evaluate_linear_model(Xs: pd.DataFrame, y_variable: pd.Series):
    linreg = linear_model.LinearRegression()
    linreg.fit(Xs, y_variable)
    yhat = linreg.predict(Xs)
    exp_var = explained_variance_score(yhat, y_variable)

    print("Explained var:", exp_var)
    print("RSS:", np.sum((yhat - y_variable) ** 2))
    print("Coeffs:", linreg.coef_)
    print("Intercept:", linreg.intercept_)

    return linreg


if __name__ == '__main__':
    df = price_data_preparation('train.csv')

    # step: Choosen highest R^2 variables with respect to Y (SalePrice)
    # Chosen without transforms
    primary_vars = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea']
    secondary_vars = ['SalePrice', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', ]
    comb_vars = np.append(primary_vars, secondary_vars[1:])

    # step: transform vars to reduce skewness
    old_df = df.copy()
    df = transform_house_pd(df)

    # step: testing and training split
    # warn: testing and splitting may not neccessarily yield a better answer
    # mask = np.random.rand(len(df)) < 0.9
    # df_test = df[~mask]
    # df = df[mask]

    # step: Linear regression fitting with transforms above

    # substep: LR using secondary vars as well as primary vars
    evaluate_linear_model(df[comb_vars].drop(columns=['SalePrice']), df['SalePrice'])

    # substep: LR using just the primary variables
    evaluate_linear_model(df[primary_vars].drop(columns=['SalePrice']), df['SalePrice'])

    # substep: LR using just the secondary variables
    evaluate_linear_model(df[secondary_vars].drop(columns=['SalePrice']), df['SalePrice'])
    # Do the coeffs line up with the trends we saw earlier?

    # substep: LR using primary vars, dropping outliers
    outlier_idxs = create_partial_plots(df[primary_vars], 'SalePrice', drop_rows=None)
    df_no_outliers = df.drop(df.index[outlier_idxs])
    evaluate_linear_model(df_no_outliers[primary_vars].drop(columns=['SalePrice']), df_no_outliers['SalePrice'])

    # substep: LR using primary vars, dropping specific outliers correspond to far points in PCA
    df_no_outliers = df.drop(df.index[[1298, 440, 523, 1190, 1061]])
    evaluate_linear_model(df_no_outliers[primary_vars].drop(columns=['SalePrice']), df_no_outliers['SalePrice'])

    # substep: LR using all vars, dropping outliers
    outlier_idxs = create_partial_plots(df[comb_vars], 'SalePrice', drop_rows=None)
    df_no_outliers = df.drop(df.index[outlier_idxs])
    evaluate_linear_model(df_no_outliers[comb_vars].drop(columns=['SalePrice']), df_no_outliers['SalePrice'])
    ######################################################################################
