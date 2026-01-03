from sklearn.preprocessing import KBinsDiscretizer

# KBinsDiscretizerを初期化し、5分位の等分布でビニングを行う
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

# 訓練データにフィットし、変換する
discretizer.fit(X_train)
X_train_binned = discretizer.transform(X_train)

# テストデータにも同じ変換を適用
X_test_binned = discretizer.transform(X_test)

# NumPy配列からPandasデータフレームへ変換し、列名を設定
X_train = pd.DataFrame(X_train_binned, index=X_train.index, columns=[f"{feat}_binned" for feat in features])
X_test = pd.DataFrame(X_test_binned, index=X_test.index, columns=[f"{feat}_binned" for feat in features])




