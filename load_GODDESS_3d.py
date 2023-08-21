import dgl
from dgl.data import DGLDataset
import torch
from preprocess.create_graph.create_graph_data_godess import create_graph
from tqdm import tqdm


data_dir='godess/data/'
num_test = 462
# dropped monosaccharide where some monosaccharide only appear once
# additional_drop = [221, 325, 2304, 1184, 1408, 1984, 1976, 1506, 525, 1531,
#                    746, 2000, 1053, 1328, 2114, 94, 929, 2134, 2287, 1575,
#                    501, 1034, 839, 646, 1602, 2403, 1857]
additional_drop = []
seed = 97211

import random
random.seed(seed)

class GODDESSDataset(DGLDataset):

    def __init__(self):
        super(GODDESSDataset, self).__init__(name='GlycoNMR_GODESS')

    def process(self):
        # Implement the processing logic to generate your graphs
        self.graphs = []

        self.train_test_indicator = []

        self.glycan_name = []

        C = create_graph(data_dir=data_dir,
                         num_test=num_test,
                         additional_drop=additional_drop,
                         seed=seed)

        train_test_indicator = C.generate_train_test_split()

        for i in tqdm(range(len(C.files_labels_list))):
            f = C.files_labels_list[i]
            if train_test_indicator[i] == 1:
                temp_g = C.create_single_graph(f, in_train_set=False, in_test_set=True)
                self.train_test_indicator.append(1)
                self.glycan_name.append(f)
            elif train_test_indicator[i] == 0:
                temp_g = C.create_single_graph(f, in_train_set=True, in_test_set=False)
                self.train_test_indicator.append(0)
                self.glycan_name.append(f)


            elif train_test_indicator[i] == -1:
                print(f, 'is dropped due to rare monosaccharide appearance')
                continue

            self.graphs.append(temp_g)

    def __getitem__(self, idx):
        return self.graphs[idx], self.train_test_indicator[idx], self.glycan_name[idx]

    def __len__(self):
        return len(self.graphs)


def split_test_val(test_set, train_ratio = 0.5):
    random.shuffle(test_set)

    train_size = int(len(test_set) * train_ratio)

    valid_data_list = test_set[train_size:]

    train_data_list = test_set[:train_size]

    return train_data_list, valid_data_list
