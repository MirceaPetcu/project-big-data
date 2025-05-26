import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from scipy.stats import pearsonr


df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'preprocessed_data.csv'))

x = df.drop(['shares', 'log_shares', 'popular', 'weekday', 'channel'], axis=1).values
y = df['log_shares'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

dummy_model = DummyRegressor(strategy='quantile', quantile=0.5)
dummy_model.fit(x_train, y_train)
y_pred_dummy = dummy_model.predict(x_test)

print(f'Dummy model MSE: {mean_squared_error(y_test, y_pred_dummy)}')
print(f'Dummy model R2: {r2_score(y_test, y_pred_dummy)}')

pearson_corr, _ = pearsonr(y_test, y_pred_dummy)
print(f'Pearson correlation: {pearson_corr}')

ridge_model = Ridge(alpha=0.1)
ridge_model.fit(x_train, y_train)
y_pred_ridge = ridge_model.predict(x_test)

print(f'Ridge model MSE: {mean_squared_error(y_test, y_pred_ridge)}')
print(f'Ridge model R2: {r2_score(y_test, y_pred_ridge)}')

pearson_corr, _ = pearsonr(y_test, y_pred_ridge)
print(f'Pearson correlation: {pearson_corr}')


