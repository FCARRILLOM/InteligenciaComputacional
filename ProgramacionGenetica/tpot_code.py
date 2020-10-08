import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('winequality-red.csv', sep=';', dtype=np.float64)
features = tpot_data.drop('quality', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['quality'], random_state=None)

# Average CV score on the training set was: -0.4081481732542279
exported_pipeline = RandomForestRegressor(bootstrap=True, max_features=0.6000000000000001, min_samples_leaf=7, min_samples_split=20, n_estimators=100)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print(mean_squared_error(testing_target, results))