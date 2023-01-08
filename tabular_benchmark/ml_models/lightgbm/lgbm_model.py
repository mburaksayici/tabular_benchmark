import numpy as np
import optuna

import lightgbm as lgb
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

class OptunaObjective:
    def __init__(self, problem_type, dtrain, dvalid, categorical_feature):
        # Hold this implementation specific arguments as the fields of the class.
        self.problem_type = problem_type
        self.metric = "accuracy"
        self.dtrain = dtrain
        self.dvalid = dvalid
        self.categorical_feature = categorical_feature
        
    def __call__(self, trial):
        param = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }

        # Add a callback for pruning.
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
        gbm = lgb.train(param, self.dtrain, valid_sets=[self.dvalid], callbacks=[pruning_callback], categorical_feature=self.categorical_feature)
        trial.set_user_attr(key="params", value=param)

        preds = gbm.predict(valid_x)
        
        pred_labels = np.rint(preds)
        accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
        return accuracy




class LGBMModel:
    def __init__(self):
        ...

    def preprocess(self, df, data_params):

        categorical_feature = data_params["numerical_columns"]

        data, target = df.loc[:, df.columns != 'y'],  df.loc[:, df.columns == 'y']
        train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)

        dtrain = lgb.Dataset(train_x, label=train_y,   categorical_feature=object_columns_feat)
        dvalid = lgb.Dataset(valid_x, label=valid_y,   categorical_feature=object_columns_feat)
        
        preprocessed_data = [dtrain,dvalid, categorical_feature]

        return preprocessed_data 

    def fit_model(self, preprocessed_data, hyperparam_opt=True):
        dtrain = preprocessed_data[0]
        dvalid = preprocessed_data[1]
        categorical_feature = preprocessed_data[2]

        if hyperparam_opt: 
            study = optuna.create_study(
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize"
            )
            study.optimize(OptunaObjective(problem_type="classification", dtrain=dtrain, dvalid=dvalid), categorical_feature = categorical_feature, n_trials=20)

            print("Number of finished trials: {}".format(len(study.trials)))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: {}".format(trial.value))

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
