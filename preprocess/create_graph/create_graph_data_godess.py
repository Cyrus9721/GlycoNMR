import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import dgl
import warnings
import json

warnings.filterwarnings("ignore")


class create_graph:
    def __init__(self, data_dir='godess/data/', adj_dir='godess/adjaency_matrix/',
                 interaction_dir='godess/adjaency_matrix/',
                 out_atom_name_embed='godess/node_embedding/atom_name_embed.csv',
                 out_bound_orig_embed='godess/node_embedding/bound_orig.csv',
                 out_atom_type_embed='godess/node_embedding/atom_type.csv',
                 out_bound_AB_embed='godess/node_embedding/bound_ab.csv',
                 out_DL_embed='godess/node_embedding/bound_dl.csv',
                 out_PF_embed='godess/node_embedding/carbon_pf.csv',
                 out_monosaccharide_accurate_embed='godess/node_embedding/monosaccharide_accurate_embed.csv',
                 out_monosaccharide_simple_embed='godess/node_embedding/monosaccharide_simple_embed.csv',
                 out_me_embed='godess/node_embedding/root_me_embed.csv',
                 out_ser_embed='godess/node_embedding/root_ser_embed.csv',
                 out_s_embed='godess/node_embedding/component_s_embed.csv',
                 out_ac_embed='godess/node_embedding/component_ac_embed.csv',
                 out_gc_embed='godess/node_embedding/component_gc_embed.csv',
                 num_test=200, additional_drop=[], seed=97211):
        self.data_dir = data_dir
        self.adj_dir = adj_dir
        self.interaction_dir = interaction_dir

        # self.atom_embed_dir = atom_embed_dir
        # self.residual_embed_dir = residual_embed_dir
        # self.mono_embed_dir = mono_embed_dir

        self.num_test = num_test
        self.seed = seed

        # files used to build a graph
        self.files_labels_list = os.listdir(self.data_dir)
        self.adj_list = os.listdir(self.adj_dir)
        self.interaction_list = os.listdir(self.interaction_dir)

        # embeddings for nodes
        # self.atom_embed = pd.read_csv(self.atom_embed_dir)
        # self.residual_embed = pd.read_csv(self.residual_embed_dir)
        # self.mono_embed = pd.read_csv(self.mono_embed_dir)

        self.atom_name_embed = pd.read_csv(out_atom_name_embed)

        self.bound_orig_embed = pd.read_csv(out_bound_orig_embed)

        self.atom_type_embed = pd.read_csv(out_atom_type_embed)

        self.bound_ab_embed = pd.read_csv(out_bound_AB_embed)

        self.bound_dl_embed = pd.read_csv(out_DL_embed)

        self.carbon_pf_embed = pd.read_csv(out_PF_embed)

        self.monosaccharide_accurate_embed = pd.read_csv(out_monosaccharide_accurate_embed)

        self.monosaccharide_simple_embed = pd.read_csv(out_monosaccharide_simple_embed)

        self.me_embed = pd.read_csv(out_me_embed)

        self.ser_embed = pd.read_csv(out_ser_embed)

        self.s_embed = pd.read_csv(out_s_embed)

        self.ac_embed = pd.read_csv(out_ac_embed)

        self.gc_embed = pd.read_csv(out_gc_embed)

        # read in duplicate mono dict

        self.additional_drop = additional_drop
    def create_single_graph(self, f1, in_train_set, in_test_set):
        """
        :param f1: name of the glycan files
        :param in_train_set: whether this glycan in training set
        :param in_test_set:  whether this glycan in testing set
        :return: a dgl graph object
        """

        """
        Add different mask for carbon and hydrogen
        """
        # carbon_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        # carbon_list = ['C1', 'C11', 'C2', 'C21', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        # carbon_list = ['C1', 'C11', 'C2', 'C21', 'C3', 'C4', 'C5', 'C6',
        #                'Neup_C1','Neup_C11', 'Neup_C2', 'Neup_C21', 'Neup_C3', 'Neup_C4',
        #                'Neup_C5', 'Neup_C6', 'Neup_C7', 'Neup_C8', 'Neup_C9']

        carbon_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'Kdo_C1',
                       'Kdo_C2', 'Kdo_C3', 'Kdo_C4', 'Kdo_C5', 'Kdo_C6', 'Kdo_C7',
                       'Kdo_C8', 'Neup_C1', 'Neup_C2', 'Neup_C3', 'Neup_C4', 'Neup_C5',
                       'Neup_C6', 'Neup_C7', 'Neup_C8', 'Neup_C9']


        # hydrogen_list = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9']
        # hydrogen_list = ['H1', 'H11', 'H2', 'H3', 'H31', 'H4', 'H5', 'H51', 'H6', 'H61', 'H7', 'H8', 'H9', 'H91']
        # hydrogen_list = ['H1', 'H11', 'H2', 'H3', 'H31', 'H4', 'H5', 'H51', 'H6', 'H61',
        #                  'Neup_H2', 'Neup_H3', 'Neup_H31', 'Neup_H4', 'Neup_H5', 'Neup_H6', 'Neup_H7', 'Neup_H8',
        #                  'Neup_H9', 'Neup_H91']

        hydrogen_list = ['H1', 'H11', 'H2', 'H3', 'H4', 'H41', 'H5', 'H51', 'H6', 'H61',
                         'H7', 'Kdo_H3', 'Kdo_H31', 'Kdo_H4', 'Kdo_H5', 'Kdo_H6', 'Kdo_H7',
                         'Kdo_H8', 'Kdo_H81', 'Neup_H2', 'Neup_H3', 'Neup_H31', 'Neup_H4',
                         'Neup_H5', 'Neup_H6', 'Neup_H7', 'Neup_H8', 'Neup_H9', 'Neup_H91']

        """
        Read in labeled pdb files with corresponding adjacency matrix.
        """
        temp_df = pd.read_csv(os.path.join(self.data_dir, f1))
        temp_adj = np.array(pd.read_csv(os.path.join(self.adj_dir, 'edges_' + f1)))

        """
        Create node features, Carbon and Hydrogen masks, training masks, testing masks.
        """
        embedding_matrix = torch.zeros([len(temp_df), len(self.atom_name_embed) + \
                                        len(self.bound_ab_embed) + \
                                        len(self.bound_dl_embed) + len(self.carbon_pf_embed) + \
                                        len(self.monosaccharide_simple_embed) + \
                                        len(self.me_embed) + len(self.ser_embed) + len(self.s_embed) + \
                                        len(self.ac_embed) + len(self.gc_embed)], dtype=torch.float32)
        # embedding_matrix = torch.zeros([len(temp_df), len(self.atom_embed) + len(self.mono_embed)], dtype=torch.float32)
        # embedding_matrix = torch.zeros([len(temp_df), len(self.atom_embed)], dtype=torch.float32)

        carbon_mask = torch.zeros(len(temp_df), dtype=torch.bool)
        hydrogen_mask = torch.zeros(len(temp_df), dtype=torch.bool)

        # temp_atom_list = temp_df['Atom_name'].values
        # temp_residual_list = temp_df['Residual_num'].values.astype(str)
        # temp_mono_list = temp_df['Residual_name'].values
        temp_label_list = temp_df['main_ring_shift'].values

        # atom_name_list = temp_df['Atom_name'].values
        atom_name_list = temp_df['New_Atom_name'].values

        bound_orig_list = temp_df['Bound'].values

        atom_type_list = temp_df['Residual_name'].values

        ab_list = temp_df['bound_AB'].values

        dl_list = temp_df['fischer_projection_DL'].values

        pf_list = temp_df['carbon_number_PF'].values

        mono_accurate_list = temp_df['Residual_accurate_name'].values

        mono_simple_list = temp_df['reformulated_standard_mono'].values

        me_list = temp_df['Me_min_atom_distance_threshold'].values

        ser_list = temp_df['Ser_atom_distance_threshold'].values

        s_list = temp_df['S_min_atom_distance_threshold'].values

        ac_list = temp_df['Ac_min_atom_distance_threshold'].values

        gc_list = temp_df['Gc_min_atom_distance_threshold'].values


        for i in range(len(temp_df)):

            # c_atom = temp_atom_list[i]
            # c_redisual = temp_residual_list[i]
            # c_mono = temp_mono_list[i]

            c_atom_name = atom_name_list[i]
            c_bound_orig = bound_orig_list[i]
            c_atom_type = atom_type_list[i]
            c_ab = ab_list[i]
            c_dl = dl_list[i]
            c_pf = pf_list[i]
            c_mono_accurate = mono_accurate_list[i]
            c_mono_simple = mono_simple_list[i]
            c_me = me_list[i]
            c_ser = ser_list[i]
            c_s = s_list[i]
            c_ac = ac_list[i]
            c_gc = gc_list[i]

            c_atom_name_embed = self.atom_name_embed[c_atom_name]
            c_bound_ab_embed = self.bound_ab_embed[c_ab]
            c_bound_dl_embed = self.bound_dl_embed[c_dl]
            c_carbon_pf_embed = self.carbon_pf_embed[c_pf]
            c_monosaccharide_simple_embed = self.monosaccharide_simple_embed[c_mono_simple]
            c_me_embed = self.me_embed[str(c_me)]
            c_ser_embed = self.ser_embed[str(c_ser)]
            c_s_embed = self.s_embed[str(c_s)]
            c_ac_embed = self.ac_embed[str(c_ac)]
            c_gc_embed = self.gc_embed[str(c_gc)]

            embedding_matrix[i, :] = torch.tensor(np.concatenate([c_atom_name_embed,
                                                                  c_bound_ab_embed,
                                                                  c_bound_dl_embed,
                                                                  c_carbon_pf_embed,
                                                                  c_monosaccharide_simple_embed,
                                                                  c_me_embed,
                                                                  c_ser_embed,
                                                                  c_s_embed,
                                                                  c_ac_embed,
                                                                  c_gc_embed], axis=0))

            # embedding_matrix[i, :] = torch.tensor(np.concatenate([self.atom_embed[c_atom], self.mono_embed[c_mono]], axis=0))
            # embedding_matrix[i, :] = torch.tensor(self.atom_embed[c_atom].values)

            if (c_atom_name in carbon_list) and (temp_label_list[i] != -1) and (temp_label_list[i] > 10):
                carbon_mask[i] = True
            if (c_atom_name in hydrogen_list) and (temp_label_list[i] != -1.0) and (temp_label_list[i] > 0):
                hydrogen_mask[i] = True

        label = torch.tensor(temp_df['main_ring_shift'].values, dtype=torch.float32)
        Carbon_Hydrogen_mask = torch.tensor(temp_df['main_ring_shift'].values != -1.0, dtype=torch.bool)

        if in_train_set and (not in_test_set):
            train_mask = Carbon_Hydrogen_mask.clone()
            test_mask = torch.zeros(len(label), dtype=torch.bool)

            train_carbon_mask = carbon_mask
            test_carbon_mask = torch.zeros(len(label), dtype=torch.bool)

            train_hydrogen_mask = hydrogen_mask
            test_hydrogen_mask = torch.zeros(len(label), dtype=torch.bool)


        elif (not in_train_set) and in_test_set:
            train_mask = torch.zeros(len(label), dtype=torch.bool)
            test_mask = Carbon_Hydrogen_mask.clone()

            train_carbon_mask = torch.zeros(len(label), dtype=torch.bool)
            test_carbon_mask = carbon_mask

            train_hydrogen_mask = torch.zeros(len(label), dtype=torch.bool)
            test_hydrogen_mask = hydrogen_mask


        else:
            raise Exception("This graph should either in training set or testing set")

        """
        Create graph with node data from adjacency matrix
        """
        src, dst = np.nonzero(temp_adj)
        g = dgl.graph((src, dst))
        g.ndata['z'] = embedding_matrix
        g.ndata['y'] = label
        g.ndata['Carbon_Hydrogen_mask'] = Carbon_Hydrogen_mask
        g.ndata['train_mask'] = train_mask
        g.ndata['test_mask'] = test_mask

        g.ndata['train_carbon_mask'] = train_carbon_mask
        g.ndata['test_carbon_mask'] = test_carbon_mask
        g.ndata['train_hydrogen_mask'] = train_hydrogen_mask
        g.ndata['test_hydrogen_mask'] = test_hydrogen_mask

        # g.ndata['coordinate'] = torch.tensor(temp_df[['x', 'y', 'z']].values).float()
        g.ndata['pos'] = torch.tensor(temp_df[['x', 'y', 'z']].values).float()
        # g.ndata['atom_type'] = temp_atom_list
        return g

    def generate_train_test_split(self):
        np.random.seed(self.seed)
        total_g = len(self.files_labels_list)
        train_test_indicator = np.zeros(total_g)

        test_index = np.sort(np.random.choice(range(total_g), size=self.num_test, replace=False))
        train_test_indicator[test_index] = 1
        return train_test_indicator

    def create_all_graph(self):
        g = dgl.DGLGraph()
        # g = dgl.graph()
        train_test_indicator = self.generate_train_test_split()
        print('--------------------------loading NMR Graph-------------------------------')
        for i in tqdm(range(len(self.files_labels_list))):
            f = self.files_labels_list[i]
            if train_test_indicator[i] == 1:
                temp_g = self.create_single_graph(f, in_train_set=False, in_test_set=True)
            elif train_test_indicator[i] == 0:
                temp_g = self.create_single_graph(f, in_train_set=True, in_test_set=False)
            elif train_test_indicator[i] == -1:
                print(f, 'is dropped due to rare monosaccharide appearance')
                continue
            g = dgl.batch([g, temp_g])

        return g, train_test_indicator


def main():
    C = create_graph()
    large_graph = C.create_all_graph()
    print(large_graph)


if __name__ == "__main__":
    main()
