import pandas as pd
from tpot import TPOTRegressor

df = pd.read_csv('winequality-red.csv', delimiter=';')

x = df.drop(columns='quality').values
y = df['quality']

regressor = TPOTRegressor(2, 20, scoring='neg_mean_squared_error', cv=10, n_jobs=-1, verbosity=2)

regressor.fit(x, y)

regressor.export('tpot_code.py')