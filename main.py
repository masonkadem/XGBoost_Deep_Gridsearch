import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import optuna

# Assuming X and y are defined earlier in your script
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Initialize an array to store out-of-fold predictions
train_oof = np.zeros(y.shape[0])

# Define the number of boosting rounds and seed
# num_round = 1000
SEED = 1

def objective(trial):
    params = {
        'objective': trial.suggest_categorical('objective', ['binary:logistic']),
        'tree_method': trial.suggest_categorical('tree_method', ['hist']),  # Options: 'gpu_hist', 'hist'
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'eval_metric': trial.suggest_categorical('eval_metric', ['auc']),
    }

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for train_index, val_index in kf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        #num_estimators = 1
        model = xgb.train(params, dtrain, 1, [(dval, 'eval')], early_stopping_rounds=50, verbose_eval=False)

        predictions = model.predict(dval)
        train_oof[val_index] = predictions

    auc = roc_auc_score(y, train_oof)

    return auc

# Example of running the Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

best_params = study.best_trial.params
