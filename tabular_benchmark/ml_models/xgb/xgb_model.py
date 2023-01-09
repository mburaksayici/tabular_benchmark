import numpy as np
import optuna

import xgboost as xgb
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split


class OptunaObjective:
    def __init__(self, problem_type, dtrain, dvalid, raw_valid, categorical_feature):
        # Hold this implementation specific arguments as the fields of the class.
        self.problem_type = problem_type
        self.metric = "accuracy"
        self.dtrain = dtrain
        self.dvalid = dvalid
        self.raw_valid = raw_valid
        self.categorical_feature = categorical_feature

    def __call__(self, trial):
        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            # use exact for small dataset.
            "tree_method": "exact",
            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "gblinear", "dart"]
            ),
            # L2 regularization weight.
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }

        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            # defines how selective algorithm is.
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            )

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical(
                "sample_type", ["uniform", "weighted"]
            )
            param["normalize_type"] = trial.suggest_categorical(
                "normalize_type", ["tree", "forest"]
            )
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        # Add a callback for pruning.
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
        bst = xgb.train(param, self.dtrain)

        trial.set_user_attr(key="params", value=param)

        preds = bst.predict(self.dvalid)
        pred_labels = np.rint(preds)
        accuracy = sklearn.metrics.accuracy_score(self.raw_valid["y"], pred_labels)
        return accuracy


class XGBModel:
    def __init__(self):
        ...

    def preprocess(self, df, data_params):

        categorical_feature = data_params["categorical_columns"]
        target_column_name = data_params["target_column"]

        df, rule_dict = XGBModel._convert_cat_str_to_int(df, categorical_feature)

        data, target = (
            df.loc[:, df.columns != target_column_name],
            df.loc[:, df.columns == target_column_name],
        )
        train_x, valid_x, train_y, valid_y = train_test_split(
            data, target, test_size=0.25
        )

        dtrain = xgb.DMatrix(train_x, label=train_y, enable_categorical=True)
        dvalid = xgb.DMatrix(valid_x, label=valid_y, enable_categorical=True)

        preprocessed_data = {
            "dtrain": dtrain,
            "dvalid": dvalid,
            "raw_valid": {"x": valid_x, "y": valid_y},
            "categorical_feature": categorical_feature,
            "target_column": target_column_name,
        }

        return preprocessed_data

    @staticmethod
    def _convert_cat_str_to_int(df, cat_features):
        rule_dict = dict()
        for cat_feat in cat_features:
            uniques_ordered = sorted(df[cat_feat].unique().tolist())
            rule_dict[cat_feat] = rule = dict(
                zip(uniques_ordered, range(len(uniques_ordered)))
            )
            df[cat_feat] = df[cat_feat].map(rule)
        return df, rule_dict

    # df, rule_dict = convert_cat_str_to_int(df, cat_features)
    def fit_model(self, preprocessed_data, hyperparam_opt=True):
        dtrain = preprocessed_data["dtrain"]
        dvalid = preprocessed_data["dvalid"]
        categorical_feature = preprocessed_data["categorical_feature"]
        target_column_name = preprocessed_data["target_column"]
        raw_valid = preprocessed_data["raw_valid"]

        if target_column_name in categorical_feature:
            categorical_feature.remove(target_column_name)

        if hyperparam_opt:
            study = optuna.create_study(
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
                direction="maximize",
            )
            study.optimize(
                OptunaObjective(
                    problem_type="classification",
                    dtrain=dtrain,
                    dvalid=dvalid,
                    categorical_feature=categorical_feature,
                    raw_valid=raw_valid,
                ),
                n_trials=100,
            )

            print("Number of finished trials: {}".format(len(study.trials)))

            print("Best trial:")
            trial = study.best_trial

            print("Value: {}".format(trial.value))

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

            return {"metric": trial.value, "params": trial.params}
