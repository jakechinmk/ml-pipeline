import argparse
import pandas as pd
import sweetviz as sv
import yaml

from pathlib import Path
from typing import Dict

class EDA:
    @staticmethod
    def preprocess(df:pd.DataFrame):
        drop_col = 'indicator'
        mask = df.loc[:, drop_col] == 'train'
        train_df = df.loc[mask, :].copy(deep=True).drop(columns=[drop_col])
        test_df = df.loc[~mask, :].copy(deep=True).drop(columns=[drop_col])
        return train_df, test_df
    
    def analysis(self, exploration_mode:str):
        target = self.config.get('pycaret').get('setup').get('target')
        save_path = self.config.get('overall').get('exploration_path')
        
        if exploration_mode == 'raw':
            df = pd.read_csv(self.config.get('overall').get('input_path'))
            report = sv.analyze(source=[df, 'raw'], 
                                target_feat=target,
                                )
        # elif exploration_mode == 'raw-compare-target':
        #     df = pd.read_csv(self.config.get('overall').get('input_path'))
        #     indicator = df.loc[:, target].unique()[0]
        #     mask = df.loc[:, target] == indicator
        #     report = sv.compare([df.loc[mask, :], 'Target = {}'.format(indicator)],
        #                         [df.loc[~mask, :], 'Target != {}'.format(indicator)],
        #                         target_feat=target
        #                         )
        elif exploration_mode == 'processed':
            df = pd.read_csv(self.config.get('overall').get('data_path'))
            train_df, test_df = self.preprocess(df)
            report = sv.compare([train_df, 'Train Data'], [test_df, 'Test Data'],
                                target_feat=target)
        report.show_html(save_path.format(exploration_mode))


    def __init__(self, config:Dict) -> None:
        self.config = config



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file path')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    pipeline = EDA(config=config)
    pipeline.analysis(exploration_mode='raw')
    pipeline.analysis(exploration_mode='processed')
