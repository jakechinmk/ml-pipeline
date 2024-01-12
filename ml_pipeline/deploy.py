import argparse
import shutil
import os
import pandas as pd
import yaml

from pycaret.classification import *
from typing import Dict

class Deployment:
    def pycaret_setup(self):
        # df = pd.read_csv(self.overall_config.get('data_path'))
        df = pd.read_csv(self.overall_config.get('input_path'))
        if self.overall_config.get('preprocess'):
            # this is to get back the original preprocess pkl from preprocessing script
            self.exp = self.exp.load_experiment(self.overall_config.get('experiment_path'),
                                                data=df,
                                                preprocess_data=True                                     
                                                )
        else:
            # this is use when data is already preprocess without using preprocess module
            # the inference pipeline will not work if the data is preprocessed by user
            # user will always need to input the data in this sense
            self.exp.setup(data=df, preprocess=False)
            self.exp.save_experiment(self.overall_config.get('experiment_path'))
    
    def create_deploy_config(self):
        model = self.exp.load_model(self.model_config.get('method'))
        model_name = model.named_steps.get('trained_model')
        self.exp.create_api(model, self.model_config.get('method'))
        self.exp.create_docker(model, self.model_config.get('method'))
        shutil.copy2(f"./{self.model_config.get('method')}.pkl", f"./deployment/{self.model_config.get('method')}.pkl")
        os.rename(f"./{self.model_config.get('method')}.py", f"./deployment/{self.model_config.get('method')}.py")
        os.rename(f"./Dockerfile", "./deployment/Dockerfile")
        

    def __init__(self, config:Dict):
        self.exp = ClassificationExperiment()
        self.pycaret_config = config.get('pycaret')
        self.overall_config = config.get('overall')
        self.model_config = self.pycaret_config.get('model')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file path')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    pipeline = Deployment(config=config)
    pipeline.pycaret_setup()
    pipeline.create_deploy_config()
