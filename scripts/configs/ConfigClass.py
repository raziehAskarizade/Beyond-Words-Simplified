# Fardin Rastakhiz @ 2023

import json
from os import path


class Config:

    def __init__(self, project_root_path: str, config_local_path: str = ''):
        self.root = project_root_path

        if config_local_path == '':
            config_local_path = 'scripts/configs/Config.json'
        config_path = path.join(self.root, config_local_path)

        with open(config_path, 'rt') as cf:
            config_data = json.load(cf)

        self.device = config_data['device'] if 'device' in config_data else 'cpu'

        # farsi
        if 'fa' in config_data:
            self.fa: FaConfig = FaConfig(config_data['fa'])

        if 'datasets' in config_data:
            self.datasets = Datasets(config_data['datasets'])

# farsi
class FaConfig:
    def __init__(self, json_data: dict):
        if 'pipeline' in json_data:
            self.pipeline: str = json_data['pipeline']


class Datasets:
    def __init__(self, json_data: dict):
        if 'amazon_review' in json_data:
            self.amazon_review = Dataset(json_data['amazon_review'])


class Dataset:
    def __init__(self, json_data: dict):
        if 'train' in json_data:
            self.train = json_data['train']
        if 'validation' in json_data:
            self.validation = json_data['validation']
        if 'test' in json_data:
            self.test = json_data['test']