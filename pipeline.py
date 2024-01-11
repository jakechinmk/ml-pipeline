import argparse
import pandas as pd
import yaml

from pycaret.classification import ClassificationExperiment
from typing import Dict
from pathlib import Path

from ml_pipeline import *

class Pipeline:
    @staticmethod
    def read_config(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def preprocess(self):
        prep = Preprocessing(config=self.config)
        prep.etl()
    
    def exploration(self):
        eda = EDA(config=self.config)
        eda.analysis(exploration_mode='raw')
        eda.analysis(exploration_mode='processed')

    def inference(self):
        inf = Inference(config=self.config)
        inf.pycaret_setup()
        inf.predict()

    def train(self):
        model = Model(config=self.config)
        model.pycaret_setup()
        model.train()
        
    def validate(self):
        val = Validator(config=config)
        validate.pycaret_setup()
        validate.explainer()

    def __init__(self, config_path:Path) -> None:
        self.config = self.read_config(config_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('function', help='function to run (preprocess, exploration, train, inference)')
    parser.add_argument('--config', help='config file path')
    args = parser.parse_args()
    
    pipeline = Pipeline(config_path=args.config)

    if args.function == 'preprocess':
        pipeline.preprocess()
    elif args.function == 'exploration':
        pipeline.exploration()
    elif args.function == 'train':
        pipeline.train()
    elif args.function == 'inference':
        pipeline.inference()
    elif args.function == 'test':
        pipeline.validate()
    else:
        print('Please provide the correct function')
    print(f'done running {args.function}')