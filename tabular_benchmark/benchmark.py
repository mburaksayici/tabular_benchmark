

import pandas as pd 
import json 

data = "data/bank/bank-full.csv"

with open('data/bank/params.json') as json_file:
    data_params = json.load(json_file)



df = pd.read_csv(data, sep = ';')

object_columns_feat = cat_features = df.select_dtypes(include='object').columns.tolist()



def convert_cat_str_to_int(df,cat_features):
    rule_dict = dict()
    for cat_feat in cat_features:
        uniques_ordered = sorted(df[cat_feat].unique().tolist())
        rule_dict[cat_feat] = rule = dict(zip(uniques_ordered,range(len(uniques_ordered))))
        df[cat_feat] = df[cat_feat].map(rule)
    return df, rule_dict




df, rule_dict = convert_cat_str_to_int(df,cat_features)


from ml_models.lightgbm.lgbm_model import LGBMModel

model = LGBMModel()


preprocessed_data = model.preprocess(df, data_params)


model.fit_model(preprocessed_data=preprocessed_data, hyperparam_opt=True)