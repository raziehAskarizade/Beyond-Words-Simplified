# Omid Davar @ 2023

from copy import copy
import numpy as np
from os import path
from pathlib import Path
from typing import Dict

import pandas as pd
from torch_geometric.loader import DataLoader

from scripts.graph_constructors.CoOccurrenceGraphConstructor import CoOccurrenceGraphConstructor
from scripts.graph_constructors.SequentialGraphConstructor import SequentialGraphConstructor
from scripts.graph_constructors.GraphConstructor import GraphConstructor, TextGraphType
from scripts.graph_data_modules.GraphDataModule import GraphDataModule
from torch.utils.data.dataset import random_split, Subset
import torch
from scripts.datasets.GraphConstructorDataset import GraphConstructorDataset, GraphConstructorDatasetRanged
from scripts.configs.ConfigClass import Config


class AGGraphDataModule(GraphDataModule):

    def __init__(self, config: Config, test_size=0.2, val_size=0.2, num_workers=2, drop_last=True, train_data_path='', test_data_path='', graphs_path='', batch_size = 32, device='cpu', shuffle = False,start_data_load=0, end_data_load=-1, graph_type: TextGraphType = TextGraphType.FULL, load_preprocessed_data = True, reweights={},removals=[], *args, **kwargs):

        super(AGGraphDataModule, self)\
            .__init__(config, device, test_size, val_size, *args, **kwargs)

        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.graph_type = graph_type
        self.reweights = reweights
        self.removals = removals
        self.graphs_path = graphs_path if graphs_path!='' else 'data/GraphData/AG'
        self.train_data_path = 'data/AG/train.csv' if train_data_path == '' else train_data_path
        self.test_data_path = 'data/AG/test.csv' if test_data_path == '' else test_data_path
        self.labels = None
        self.dataset = None
        self.shuffle = shuffle
        self.start_data_load = start_data_load
        self.end_data_load = end_data_load
        self.df: pd.DataFrame = pd.DataFrame()
        self.__train_dataset, self.__val_dataset, self.__test_dataset = None, None, None
        self.load_preprocessed_data = load_preprocessed_data
        
        
    def load_labels(self):
        self.train_df = pd.read_csv(path.join(self.config.root, self.train_data_path))
        self.test_df = pd.read_csv(path.join(self.config.root, self.test_data_path))
        self.train_df.columns = ['Class', 'Title', 'Description']
        self.test_df.columns = ['Class', 'Title', 'Description']
        self.train_df['Description'] = self.train_df['Title'].astype(str) + ' ' +  self.train_df['Description'].astype(str)
        self.test_df['Description'] = self.test_df['Title'].astype(str) + ' ' +  self.test_df['Description'].astype(str)
        self.train_df = self.train_df[['Class', 'Description']]
        self.test_df = self.test_df[['Class', 'Description']]
        self.df = pd.concat([self.train_df, self.test_df])
        self.end_data_load = self.end_data_load if self.end_data_load>0 else self.df.shape[0]
        self.end_data_load = self.end_data_load if self.end_data_load < self.df.shape[0] else self.df.shape[0] 
        self.df = self.df.iloc[self.start_data_load:self.end_data_load]
        self.df.index = np.arange(0, self.end_data_load - self.start_data_load)
        # activate one line below
        labels = self.df['Class'][:self.end_data_load - self.start_data_load]
        labels = labels.to_numpy()
        labels = torch.from_numpy(labels)
        self.num_classes = len(torch.unique(labels))
        self.labels = torch.nn.functional.one_hot((labels-1).to(torch.int64)).to(torch.float32).to(self.device)
        
        self.num_data = self.df.shape[0]
        self.train_range = range(int((1-self.val_size-self.test_size)*self.num_data))
        self.val_range = range(self.train_range[-1]+1, int((1-self.test_size)*self.num_data))
        self.test_range = range(self.val_range[-1]+1, self.num_data)
        
        # graph_constructor = self.graph_constructors[TextGraphType.CO_OCCURRENCE]
        
    def load_graphs(self):
        self.graph_constructors = self.__set_graph_constructors(self.graph_type)
        
        self.dataset, self.num_node_features = {}, {}
        self.__train_dataset, self.__val_dataset, self.__test_dataset = {}, {}, {}
        self.__train_dataloader, self.__test_dataloader, self.__val_dataloader = {}, {}, {}
        for key in self.graph_constructors:
            self.graph_constructors[key].setup(self.load_preprocessed_data)
            # reweighting
            if key in self.reweights:
                for r in self.reweights[key]:
                    self.graph_constructors[key].reweight_all(r[0] , r[1])
                    
            self.dataset[key] = GraphConstructorDatasetRanged(self.graph_constructors[key], self.labels , self.start_data_load , self.end_data_load)
            
            self.__train_dataset[key] = Subset(self.dataset[key], self.train_range)
            self.__val_dataset[key] = Subset(self.dataset[key], self.val_range)
            self.__test_dataset[key] = Subset(self.dataset[key], self.test_range)
                        
            self.__train_dataloader[key] =  DataLoader(self.__train_dataset[key], batch_size=self.batch_size, drop_last=self.drop_last, shuffle=self.shuffle, num_workers=0, persistent_workers=False)
            self.__test_dataloader[key] =  DataLoader(self.__test_dataset[key], batch_size=self.batch_size, num_workers=0, persistent_workers=False)
            self.__val_dataloader[key] =  DataLoader(self.__val_dataset[key], batch_size=self.batch_size, num_workers=0, persistent_workers=False)
            
        self.set_active_graph(key)
        
    
    def update_batch_size(self, batch_size):
        self.batch_size = batch_size
        
        for key in self.graph_constructors:
            self.__train_dataloader[key] =  DataLoader(self.__train_dataset[key], batch_size=self.batch_size, drop_last=self.drop_last, shuffle=self.shuffle, num_workers=0, persistent_workers=False)
            self.__test_dataloader[key] =  DataLoader(self.__test_dataset[key], batch_size=self.batch_size, num_workers=0, persistent_workers=False)
            self.__val_dataloader[key] =  DataLoader(self.__val_dataset[key], batch_size=self.batch_size, num_workers=0, persistent_workers=False)
            
        self.set_active_graph(key)    
    
    def get_data(self, datamodule):
        self.labels = datamodule.labels
        self.num_classes = datamodule.num_classes
        self.graph_constructors = datamodule.graph_constructors
        self.dataset, self.num_node_features = datamodule.dataset, datamodule.num_node_features
        self.__train_dataset, self.__val_dataset, self.__test_dataset = datamodule.__train_dataset, datamodule.__val_dataset, datamodule.__test_dataset
        self.__train_dataloader, self.__test_dataloader, self.__val_dataloader = datamodule.__train_dataloader, datamodule.__test_dataloader, datamodule.__val_dataloader
        self.set_active_graph(datamodule.active_key )
        
    def set_active_graph(self, graph_type: TextGraphType = TextGraphType.CO_OCCURRENCE):
        assert graph_type in self.dataset, 'The provided key is not valid'
        self.active_key = graph_type
        sample_graph = self.graph_constructors[self.active_key].get_first()
        self.num_node_features = sample_graph.num_features
           
    def create_sub_data_loader(self, begin: int, end: int):
        for key in self.graph_constructors:            
            dataset = GraphConstructorDatasetRanged(self.graph_constructors[key], self.labels, begin, end)
            
            train_dataset = Subset(dataset, self.train_range)
            val_dataset = Subset(dataset,  self.val_range)
            test_dataset= Subset(dataset,  self.test_range)
                
            self.__train_dataloader[key] =  DataLoader(train_dataset, batch_size=self.batch_size, drop_last=self.drop_last, shuffle=self.shuffle, num_workers=0, persistent_workers=False)
            self.__test_dataloader[key] =  DataLoader(test_dataset, batch_size=self.batch_size, num_workers=0, persistent_workers=False)
            self.__val_dataloader[key] =  DataLoader(val_dataset, batch_size=self.batch_size, num_workers=0, persistent_workers=False)
            
        self.set_active_graph(key)
        
    def prepare_data(self):
        pass
        
    def setup(self, stage: str):
        pass

    def teardown(self, stage: str) -> None:
        pass

    def train_dataloader(self):
        return self.__train_dataloader[self.active_key ]

    def test_dataloader(self):
        return self.__test_dataloader[self.active_key ]

    def val_dataloader(self):
        return self.__val_dataloader[self.active_key ]

    def __set_graph_constructors(self, graph_type: TextGraphType):
        graph_type = copy(graph_type)
        graph_constructors: Dict[TextGraphType, GraphConstructor] = {}
        if TextGraphType.CO_OCCURRENCE in graph_type:
            graph_constructors[TextGraphType.CO_OCCURRENCE] = self.__get_co_occurrence_graph()
            graph_type = graph_type - TextGraphType.CO_OCCURRENCE
              
        if TextGraphType.SEQUENTIAL in graph_type:
            graph_constructors[TextGraphType.SEQUENTIAL] = self.__get_sequential_graph()
            graph_type = graph_type - TextGraphType.SEQUENTIAL
            
            
        return graph_constructors

    def __get_co_occurrence_graph(self):
        Path(path.join(self.graphs_path, 'co_occ')).mkdir(parents=True, exist_ok=True)
        return CoOccurrenceGraphConstructor(self.df['Description'][:self.end_data_load], path.join(self.graphs_path, 'co_occ'), self.config, load_preprocessed_data=True, naming_prepend='graph', start_data_load=self.start_data_load, end_data_load=self.end_data_load)
    
    def __get_sequential_graph(self):
        Path(path.join(self.graphs_path, 'seq_gen')).mkdir(parents=True, exist_ok=True)
        return SequentialGraphConstructor(self.df['Description'][:self.end_data_load], path.join(self.graphs_path, 'seq_gen'), self.config, load_preprocessed_data=True, naming_prepend='graph', start_data_load=self.start_data_load, end_data_load=self.end_data_load , use_general_node=True)
       
    def zero_rule_baseline(self):
        return f'zero_rule baseline: {(len(self.labels[self.labels>0.5])* 100.0 / len(self.labels))  : .2f}%'