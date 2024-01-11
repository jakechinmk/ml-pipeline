import argparse
import pandas as pd
import yaml

from pycaret.classification import ClassificationExperiment
from typing import Dict

class Model:
    @staticmethod
    def preprocess(df:pd.DataFrame):
        drop_col = 'indicator'
        mask = df.loc[:, drop_col] == 'train'
        train_df = df.loc[mask, :].copy(deep=True).drop(columns=[drop_col])
        test_df = df.loc[~mask, :].copy(deep=True).drop(columns=[drop_col])
        return train_df, test_df

    def pycaret_setup(self):
        df = pd.read_csv(self.overall_config.get('data_path'))
        if self.overall_config.get('preprocess'):
            # this is to get back the original preprocess pkl from preprocessing script
            train_df, test_df = self.preprocess(df)
            self.exp = self.exp.load_experiment(self.overall_config.get('experiment_path'),
                                                data=train_df,
                                                test_data=test_df,
                                                preprocess_data=False                                     
                                                )
        else:
            # this is use when data is already preprocess without using preprocess module
            # the inference pipeline will not work if the data is preprocessed by user
            # user will always need to input the data in this sense
            self.exp.setup(data=df, preprocess=False)
            self.exp.save_experiment(self.overall_config.get('experiment_path'))
    
    def train(self):
        model_config = self.pycaret_config.get('model')

        method = model_config.get('method')
        n_select = model_config.get('n_select')
        metric = model_config.get('metric')
        tune = model_config.get('tune')
        ensemble = model_config.get('ensemble')
        blend = model_config.get('blend')
        stack = model_config.get('stack')

        if model_config.get('method') == 'compare':
            model = self.exp.compare_models(n_select=n_select, sort=metric)
        else:
            model = self.exp.create_model(model_config.get('method'),)

        if isinstance(model, list):
            if tune:
                tune_model = [self.exp.tune_model(i) for i in model]
            if ensemble:
                bagged_model = [self.exp.ensemble_model(i) for i in tune_model]
            if blend:
                blender = self.exp.blend_models(model)
            if stack:
                stacker = self.exp.stack_models(model)
        else:
            if tune:
                tune_model = self.exp.tune_model(model)
            # wouldn't able to ensemble, blend, stack

        best = self.exp.automl(optimize=metric)

        self.exp.save_experiment(self.overall_config.get('experiment_path'))
        self.exp.save_model(best, model_config.get('method'))

    def __init__(self, config:Dict) -> None:
        self.exp = ClassificationExperiment()
        self.pycaret_config = config.get('pycaret')
        self.overall_config = config.get('overall')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file path')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    pipeline = Model(config=config)
    pipeline.pycaret_setup()
    pipeline.train()