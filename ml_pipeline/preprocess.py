import argparse
import pandas as pd
import yaml

from pathlib import Path
from pycaret.classification import ClassificationExperiment
from typing import Dict

class Preprocessing:
    # def clean(self, df:pd.DataFrame):
    #     config = self.preprocessing_config

    #     # drop any columns that having missing values more than the threshold
    #     temp_series = df.isna().sum() / df.shape[0]
    #     mask = temp_series > config.get('missing').get('threshold')
    #     drop_col_list = temp_series[mask].index.tolist()
    #     result_df = df.drop(columns=drop_col_list)

    #     # drop columns that only having one categories
    #     temp_series = result_df.nunique() 
    #     mask1 = temp_series == 1 # drop one categories
    #     mask2 = temp_series == result_df.shape[0] # drop all unique index
    #     drop_col_list = temp_series[mask1 | mask2].index.tolist()
    #     result_df = df.drop(columns=drop_col_list)
    #     return result_df

    def etl(self):
        df = pd.read_csv(self.overall_config.get('input_path'))
        # df = self.clean(df=df)
        self.exp.setup(data=df, **self.pycaret_config.get('setup'))
        self.exp.save_experiment(self.overall_config.get('experiment_path'))

        # extra handling to get the output
        df_train_transformed = self.exp.get_config('train_transformed')
        df_train_transformed.loc[:, 'indicator'] = 'train'
        
        df_test_transformed = self.exp.get_config('test_transformed')
        df_test_transformed.loc[:, 'indicator'] = 'test'

        df_transformed = pd.concat([df_train_transformed, df_test_transformed],
                                   axis=0
                                   )
        df_transformed.to_csv(self.overall_config.get('output_path'), index=False)

    def __init__(self, config:Dict) -> None:
        self.exp = ClassificationExperiment()
        self.pycaret_config = config.get('pycaret')
        self.preprocessing_config = config.get('preprocessing')
        self.overall_config = config.get('overall')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file path')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print('loaded config file') # need to change to log later
    pipeline = Preprocessing(config=config)
    pipeline.etl()