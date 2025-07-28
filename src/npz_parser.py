from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Callable, List
import os.path as osp

import numpy as np 
import torch
import shutil
import os
import copy
import random
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from utils.data_utils import read_npz_file
from parser_func import *

class NpzParser():
    '''
        Parse the npz file into an inmemory torch_geometric.data.Data object
    '''
    def __init__(self, data_dir, circuit_path, label_path, \
                 random_shuffle=False, trainval_split=0.9): 
        self.data_dir = data_dir
        self.dataset = self.inmemory_dataset(data_dir, circuit_path, label_path)
        self.trainval_split = trainval_split
        self.random_shuffle = random_shuffle
        # if random_shuffle:
        #     perm = torch.randperm(len(dataset))
        #     dataset = dataset[perm]
        
        # self.train_dataset = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        # self.val_dataset = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    def get_dataset(self, split_with_design=False):
        
        data_len = len(self.dataset)
        
        if not split_with_design:
            if self.random_shuffle:
                perm = torch.randperm(len(self.dataset))
                self.dataset = self.dataset[perm]
            training_cutoff = int(data_len * self.trainval_split)
            train_dataset = self.dataset[:training_cutoff]
            val_dataset = self.dataset[training_cutoff:]  
        else:
            module_name_set = set()
            for i in range(len(self.dataset)):
                module_name_set.add(self.dataset[i].name.split('_trace')[0])

            module_name_list = sorted(module_name_set)
            random.seed(666)    # the same shuffle each time
            random.shuffle(module_name_list)
            select_modules = module_name_list[:int(len(module_name_set) * self.trainval_split)]

            sel_mask = []
            if self.random_shuffle:
                perm = torch.randperm(len(self.dataset))
                self.dataset = self.dataset[perm]
            for i in range(len(self.dataset)):
                module_name = self.dataset[i].name.split('_trace')[0]
                if module_name in select_modules:
                    sel_mask.append(True)
                else:
                    sel_mask.append(False)

            train_dataset = self.dataset[np.array(sel_mask)]
            val_dataset = self.dataset[~np.array(sel_mask)]  
        
        return train_dataset, val_dataset

                
    
    class inmemory_dataset(InMemoryDataset):
        def __init__(self, root, graph_path, label_path, transform=None, pre_transform=None, pre_filter=None):
            self.name = 'npz_inmm_dataset'
            self.root = root
            self.graph_path = graph_path
            self.label_path = label_path
            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0])
        
        @property
        def raw_dir(self):
            return self.root

        @property
        def processed_dir(self):
            name = 'inmemory-' + os.path.basename(self.graph_path).split('.')[0]
            return osp.join(self.root, name)

        @property
        def raw_file_names(self) -> List[str]:
            return [self.graph_path, self.label_path]

        @property
        def processed_file_names(self) -> str:
            return ['data.pt']

        def download(self):
            pass
        
        def process(self):
            data_list = []
            designs = read_npz_file(self.graph_path)['designs'].item()
            labels = read_npz_file(self.label_path)['labels'].item()
            
            for design_idx, design_name in enumerate(designs):
                print('Parse design: {}, {:} / {:} = {:.2f}%'.format(design_name, design_idx, len(designs), design_idx / len(designs) * 100))
                x = designs[design_name]['x']  # x[i][0], node_id; x[i][1], node_type; x[i][2], node_width
                edge_index = designs[design_name]['edge_index']
                edge_type = designs[design_name]['edge_type']
                sim_res = designs[design_name]['sim_res']
                has_sim_res = designs[design_name]['has_sim_res']
                
                power = designs[design_name]['power']
                slack = designs[design_name]['slack']
                area = designs[design_name]['area']
                
                if power == None or slack == None or area == None or abs(slack) < 1e-5 or abs(area) < 1e-5 or abs(power[0]) < 1e-5:
                    continue
                    
                y = labels[design_name]['y']
                
                # added by XXXX-5 2024/09/15
                # using each simulation result as a separate data
                for trace_id, trace in enumerate(sim_res):
                    graph = parse_pyg_mlpgate(
                        x=x, edge_index=edge_index, edge_type=edge_type, sim_res=trace, has_sim_res=has_sim_res, y=y,
                        power=power[trace_id], slack=slack, area=area
                    )
                    graph.name = design_name + '_trace' + str(trace_id)
                    data_list.append(graph)

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
            print('Total Designs: {:}'.format(len(data_list)))

        def __repr__(self) -> str:
            return f'{self.name}({len(self)})'