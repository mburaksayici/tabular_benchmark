import sys
import inspect

import pandas as pd
from tabulate import tabulate
from pprint import pprint
import data
import ml_models
from data import DataReader


class TabularBenchmark:
    MODEL_CLASSES = dict(inspect.getmembers(ml_models, inspect.isclass))
    DATASET_CLASSES = dict(inspect.getmembers(data, inspect.isclass))

    def __init__(self, datasets=["BankDataset"], models=["XGBModel", "LGBMModel"]):
        self.models = models
        self.datasets = datasets

    def register_model():
        pass

    def register_dataset():
        pass

    def _prepare_data(self, dataset_cls):
        data_reader = DataReader(dataset_cls)
        df, params = data_reader.read()
        return df, params

    def create_benchmark(self):
        results = dict()
        params = dict()

        for dataset in self.datasets:
            # Read data
            dataset_cls = self.DATASET_CLASSES[dataset]
            df, data_params = self._prepare_data(dataset_cls)

            results[dataset] = dict()
            params[dataset] = dict()

            for model in self.models:
                model_cls = self.MODEL_CLASSES[model]()
                preprocessed_data = model_cls.preprocess(df, data_params)
                benchmark = model_cls.fit_model(
                    preprocessed_data=preprocessed_data, hyperparam_opt=True
                )

                results[dataset][model] = benchmark["metric"]
                params[dataset][model] = benchmark["params"]
        return results, params


tabular_benchmark = TabularBenchmark(
    datasets=["BankDataset"], models=["XGBModel", "LGBMModel"]
)
# tabular_benchmark.register_model()
# tabular_benchmark.register_dataset()

results, params = tabular_benchmark.create_benchmark()
results_df = pd.DataFrame.from_records(results)


print(
    tabulate(
        results_df,
        headers=results_df.columns,
        floatfmt=".5f",
        showindex=True,
        tablefmt="psql",
    )
)

pprint(params)
