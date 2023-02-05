xgr = xtrain.groupby("breath_id")["u_in"]
xtrain["last_value_u_in"] = xgr.transform("last")
xtrain["u_in_cumsum"] = xgr.cumsum()
xtrain["u_in_lag1"] = xgr.shift(1)
xtrain["u_in_lag_back1"] = xgr.shift(-1)
xtrain.fillna(0, inplace=True)


xtrain["R"] = xtrain["R"].astype(str)
xtrain["C"] = xtrain["C"].astype(str)
xtrain = pd.get_dummies(xtrain)

del xgr

xtrain.head(3)

print(xtrain.shape)
