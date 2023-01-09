import sys
import os
import json
import inspect

import pandas as pd


class DataReader:
    """
    Class that reads any tabular data and returns it as a pandas dataframe
    """

    def __init__(self, dataset_cls):
        self.dataset_cls = dataset_cls

    def _preprocess(
        self,
    ):
        pass

    def read(self):
        # Get filename from dataset class
        dataset_file_name = self.dataset_cls.DATASET_FILE
        params_file_name = self.dataset_cls.PARAMS_JSON
        # Get absolute filepath
        dirname = os.path.dirname(inspect.getfile(self.dataset_cls))
        absolute_dataset_file_name = os.path.join(dirname, dataset_file_name)
        absolute_params_file_name = os.path.join(dirname, params_file_name)

        # TO DO : Extend type support for csv and others
        df = pd.read_csv(absolute_dataset_file_name, sep=self.dataset_cls.DELIM)
        # Read params
        f = open(absolute_params_file_name)
        params = json.load(f)

        return df, params
