import argparse
import pandas as pd
import yaml

from pycaret.classification import ClassificationExperiment
from typing import Dict

class Inference:
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

    def predict(self):
        df = pd.read_csv(self.overall_config.get('inference_path'))
        target = self.pycaret_config.get('setup').get('target')
        if target in df.columns:
            df.drop(columns=target, inplace=True)

        model = self.exp.load_model(self.model_config.get('method'))
        df_prediction = self.exp.predict_model(model, data=df)
        col_list = ['prediction_score']
        df_prediction = df_prediction.loc[:, col_list]
        df_prediction.rename(columns={
            'prediction_score':'Probability',
        }, inplace=True)
        df_prediction.loc[:, 'Id'] = df_prediction.index
        df_prediction.to_csv(self.overall_config.get('predict_path'), index=False)
    
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
    
    pipeline = Inference(config=config)
    pipeline.pycaret_setup()
    pipeline.predict()