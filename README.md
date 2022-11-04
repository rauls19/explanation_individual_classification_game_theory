# Implementation of the algorithm from the paper called "An Efficient Explanation of Individual Classifications using Game Theory".

## Models supported
- LinearRegression
- MLPRegressor
- XGBoost Regressor

## How to use it

- Initialise model
- Initialise miniCoalitionalGame 
- (Optional) Check your model compatibility
- Execute the explainer (automatically checks if the model is available)

### Examples

#### Liner Model
```
lm_model = LinearRegression()
minicg_lm = miniCoalitionalGame()
if not minicg_lm.check_model_availability(lm_model):
    print('Model is not available')
print('Linear Regression: ', minicg_lm.explainerContribution(lm_model, X, Y, X, 20, ['Age', 'Diastolic BP']))
```

#### MLPRegressor
```
nn_reg = MLPRegressor(hidden_layer_sizes=(60, 20),  activation='logistic', solver='adam', 
alpha=0.01, batch_size='auto', learning_rate='constant', learning_rate_init=0.01, max_iter=1000,
    shuffle=False, tol=0.0001, verbose=False, early_stopping= True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
coalgame_nn_reg = miniCoalitionalGame()
print('MLP Regressor: ', coalgame_nn_reg.explainerContribution(nn_reg, X, Y, X, 5, ['Age']))
```

#### XGBoost Regressor
```
xgbm = xgb.XGBRegressor(learning_rate =0.01, n_estimators=215, max_depth=10, min_child_weight=0.8, subsample=1, nthread=4)
xgbm = xgb.XGBRegressor(learning_rate =0.01, n_estimators=215, max_depth=10, min_child_weight=0.8, subsample=1, nthread=4)
coalgame_xgb = miniCoalitionalGame()
print('XGB Regressor: ', coalgame_xgb.explainerContribution(xgbm, X, Y, X, 5, ['Age', 'Diastolic BP']))
```
