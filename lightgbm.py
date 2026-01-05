standard_large_lgbm_params = {
    "n_estimators": 20000,
    "learning_rate": 0.001,
    "max_depth": 6,
    "num_leaves": 2**6,
    "colsample_bytree": 0.1,
}

# https://docs.numer.ai/numerai-tournament/models
deep_lgbm_params = {
  "n_estimators": 30000,
  "learning_rate": 0.001,
  "max_depth": 10,
  "num_leaves": 1024,
  "colsample_bytree": 0.1,
  "min_data_in_leaf": 10000
}

# https://numer.ai/tutorial/hello-numerai
import lightgbm as lgb

model = lgb.LGBMRegressor(
  n_estimators=2000,
  learning_rate=0.01,
  max_depth=5,
  num_leaves=2**5-1,
  colsample_bytree=0.1
)

# model = lgb.LGBMRegressor(
#     n_estimators=30_000,
#     learning_rate=0.001,
#     max_depth=10,
#     num_leaves=2**10,
#     colsample_bytree=0.1
#     min_data_in_leaf=10000,
# )

model.fit(
  train[feature_set],
  train["target"]
)