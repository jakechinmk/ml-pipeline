import argparse
import pandas as pd
import yaml
import os

from pycaret.classification import ClassificationExperiment
from typing import Dict
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
class Validator:
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
            
    def explainer(self):
        model = self.exp.load_model(self.model_config.get('method')) 
        x_test = self.exp.get_config('X_test_transformed')
        y_test = self.exp.get_config('y_test_transformed')
        explainer_yaml_path = self.overall_config.get('explainer_yaml_path')
        explainer_path = self.overall_config.get('explainer_path')
        if os.path.exists(explainer_yaml_path):
            dashboard = ExplainerDashboard.from_config(explainer_path, explainer_yaml_path)
            dashboard.run()
        else:
            explainer = ClassifierExplainer(model=model,
                                            X=x_test,
                                            y=y_test
                                            )
            dashboard = ExplainerDashboard(explainer)
            dashboard.run()
            dashboard.to_yaml(filepath=explainer_yaml_path, explainerfile=explainer_path.split('/')[-1], dump_explainer=True)
        
        
    def __init__(self, config:Dict) -> None:
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
    
    pipeline = Validator(config=config)
    pipeline.pycaret_setup()
    pipeline.explainer()