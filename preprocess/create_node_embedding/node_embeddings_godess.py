import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.nn as nn


class create_node_embeddings:
    """
    generate embeddings for atom type, residual belongs and monosaccharide number
    # 1, atom name: 04, C4, C3...
    # 2, bound_orig: A, ... from original bound type
    # 3, atom_type: O, C, H, N
    # 4, ab_list: bound A, B
    # 5, dl_list: fischer d, l
    # 6, pf_list: number of carbon p, f
    # 7, mono_accurate_list: accurate monosaccharide name, a-D-Glcp
    # 8, mono_simple_list: simple monosaccharide name, glc
    # 9, me_list: Me root interatction
    # 10, ser_list: Ser root interaction
    # 11, s_list: Sulfur residue effect
    # 12, ac_list: Ac residue effect
    # 13, gc_list: Gc residue effect

    """

    def __init__(self, data_dir='godess/data/',
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
                 seed=97211):
        self.data_dir = data_dir

        self.out_atom_name_embed = out_atom_name_embed

        self.out_bound_orig_embed = out_bound_orig_embed

        self.out_atom_type_embed = out_atom_type_embed

        self.out_bound_ab_embed = out_bound_AB_embed

        self.out_bound_dl_embed = out_DL_embed

        self.out_carbon_pf_embed = out_PF_embed

        self.out_monosaccharide_accurate_embed = out_monosaccharide_accurate_embed

        self.out_monosaccharide_simple_embed = out_monosaccharide_simple_embed

        self.out_me_embed = out_me_embed

        self.out_ser_embed = out_ser_embed

        self.out_s_embed = out_s_embed

        self.out_ac_embed = out_ac_embed

        self.out_gc_embed = out_gc_embed

        self.seed = seed

    def create_all_embeddings(self, atom_name_dim=512, bound_orig_dim=32, atom_type_dim=32, ab_dim=64,
                              dl_dim=64, pf_dim=64, mono_accurate_dim=128, mono_simple_dim=256,
                              me_dim=64, ser_dim=64, s_dim=128, ac_dim=128, gc_dim=64):
        """
        :param atom_dim: atom feature dimension
        :param residual_dim: residual feature dimension
        :param mono_dim: monosaccharide feature dimension
        :return: three embedding dataframe: dimension * type
        """
        torch.manual_seed(self.seed)
        files_labels_list = os.listdir(self.data_dir)
        # 1, atom name: 04, C4, C3...
        atom_name_list = []

        # 2, A, ... from original bound type
        bound_orig_list = []

        # 3, O, C, H, N
        atom_type_list = []

        # 4, bound A, B
        ab_list = []

        # 5, fischer d, l
        dl_list = []

        # 6, number of carbon p, f
        pf_list = []

        # 7, accurate monosaccharide name, a-D-Glcp
        mono_accurate_list = []

        # 8, simple monosaccharide name, glc
        mono_simple_list = []

        # 9, Me root interatction
        me_list = []

        # 10, Ser root interaction
        ser_list = []

        # 11, Sulfur residue effect
        s_list = []

        # 12, Ac residue effect
        ac_list = []

        # 13, Gc residue effect
        gc_list = []


        for i in tqdm(range(len(files_labels_list))):
            f1 = files_labels_list[i]
            temp_df = pd.read_csv(os.path.join(self.data_dir, f1))

            # atom_name_list.extend(list(temp_df['Atom_name'].values))
            atom_name_list.extend(list(temp_df['New_Atom_name'].values))

            bound_orig_list.extend(list(temp_df['Bound'].values))

            atom_type_list.extend(list(temp_df['Residual_name'].values))

            ab_list.extend(list(temp_df['bound_AB'].values))

            dl_list.extend(list(temp_df['fischer_projection_DL'].values))

            pf_list.extend(list(temp_df['carbon_number_PF'].values))

            mono_accurate_list.extend(list(temp_df['Residual_accurate_name'].values))

            mono_simple_list.extend(list(temp_df['reformulated_standard_mono'].values))

            me_list.extend(list(temp_df['Me_min_atom_distance_threshold'].values))

            ser_list.extend(list(temp_df['Ser_atom_distance_threshold'].values))

            s_list.extend(list(temp_df['S_min_atom_distance_threshold'].values))

            ac_list.extend(list(temp_df['Ac_min_atom_distance_threshold'].values))

            gc_list.extend(list(temp_df['Gc_min_atom_distance_threshold'].values))

        """
        get the unique type 
        """
        atom_name_list = np.sort(np.unique(atom_name_list))
        bound_orig_list = np.sort(np.unique(bound_orig_list))
        atom_type_list = np.sort(np.unique(atom_type_list))
        ab_list = np.sort(np.unique(ab_list))
        dl_list = np.sort(np.unique(dl_list))
        pf_list = np.sort(np.unique(pf_list))
        mono_accurate_list = np.sort(np.unique(mono_accurate_list))
        mono_simple_list = np.sort(np.unique(mono_simple_list))

        me_list = np.sort(np.unique(me_list))
        ser_list = np.sort(np.unique(ser_list))
        s_list = np.sort(np.unique(s_list))
        ac_list = np.sort(np.unique(ac_list))
        gc_list = np.sort(np.unique(gc_list))


        atom_name_embedding = nn.Embedding(len(atom_name_list),
                                           embedding_dim=atom_name_dim).weight.clone().detach().numpy()
        df_atom_name_embedding = pd.DataFrame(atom_name_embedding.T)
        df_atom_name_embedding.columns = atom_name_list

        bound_orig_embedding = nn.Embedding(len(bound_orig_list),
                                            embedding_dim=bound_orig_dim).weight.clone().detach().numpy()
        df_bound_orig_embedding = pd.DataFrame(bound_orig_embedding.T)
        df_bound_orig_embedding.columns = bound_orig_list

        atom_type_embedding = nn.Embedding(len(atom_type_list),
                                           embedding_dim=atom_type_dim).weight.clone().detach().numpy()
        df_atom_type_embedding = pd.DataFrame(atom_type_embedding.T)
        df_atom_type_embedding.columns = atom_type_list

        ab_embedding = nn.Embedding(len(ab_list), embedding_dim=ab_dim).weight.clone().detach().numpy()
        df_ab_embedding = pd.DataFrame(ab_embedding.T)
        df_ab_embedding.columns = ab_list

        dl_embedding = nn.Embedding(len(dl_list), embedding_dim=dl_dim).weight.clone().detach().numpy()
        df_dl_embedding = pd.DataFrame(dl_embedding.T)
        df_dl_embedding.columns = dl_list

        pf_embedding = nn.Embedding(len(pf_list), embedding_dim=pf_dim).weight.clone().detach().numpy()
        df_pf_embedding = pd.DataFrame(pf_embedding.T)
        df_pf_embedding.columns = pf_list

        mono_accurate_embedding = nn.Embedding(len(mono_accurate_list),
                                               embedding_dim=mono_accurate_dim).weight.clone().detach().numpy()
        df_mono_accurate_embedding = pd.DataFrame(mono_accurate_embedding.T)
        df_mono_accurate_embedding.columns = mono_accurate_list

        mono_simple_embedding = nn.Embedding(len(mono_simple_list),
                                             embedding_dim=mono_simple_dim).weight.clone().detach().numpy()
        df_mono_simple_embedding = pd.DataFrame(mono_simple_embedding.T)
        df_mono_simple_embedding.columns = mono_simple_list

        me_embedding = nn.Embedding(len(me_list), embedding_dim=me_dim).weight.clone().detach().numpy()
        df_me_embedding = pd.DataFrame(me_embedding.T)
        df_me_embedding.columns = me_list

        ser_embedding = nn.Embedding(len(ser_list), embedding_dim=ser_dim).weight.clone().detach().numpy()
        df_ser_embedding = pd.DataFrame(ser_embedding.T)
        df_ser_embedding.columns = ser_list

        s_embedding = nn.Embedding(len(s_list), embedding_dim=s_dim).weight.clone().detach().numpy()
        df_s_embedding = pd.DataFrame(s_embedding.T)
        df_s_embedding.columns = s_list

        ac_embedding = nn.Embedding(len(ac_list), embedding_dim=ac_dim).weight.clone().detach().numpy()
        df_ac_embedding = pd.DataFrame(ac_embedding.T)
        df_ac_embedding.columns = ac_list

        gc_embedding = nn.Embedding(len(gc_list), embedding_dim=gc_dim).weight.clone().detach().numpy()
        df_gc_embedding = pd.DataFrame(gc_embedding.T)
        df_gc_embedding.columns = gc_list



        return df_atom_name_embedding, df_bound_orig_embedding, df_atom_type_embedding, df_ab_embedding, df_dl_embedding, df_pf_embedding, \
               df_mono_accurate_embedding, df_mono_simple_embedding, df_me_embedding, df_ser_embedding, df_s_embedding, df_ac_embedding, df_gc_embedding

    def write_all_embeddings(self, atom_name_dim=512, bound_orig_dim=32, atom_type_dim=32, ab_dim=64,
                              dl_dim=64, pf_dim=64, mono_accurate_dim=128, mono_simple_dim=256, me_dim=64, ser_dim=64, s_dim=128, ac_dim=128, gc_dim=64):
        df_atom_name_embedding, df_bound_orig_embedding, df_atom_type_embedding, df_ab_embedding, df_dl_embedding, df_pf_embedding, \
        df_mono_accurate_embedding, df_mono_simple_embedding, df_me_embedding, df_ser_embedding, df_s_embedding, df_ac_embedding, df_gc_embedding = self.create_all_embeddings(atom_name_dim, bound_orig_dim, atom_type_dim, ab_dim,
                                                                                          dl_dim, pf_dim, mono_accurate_dim, mono_simple_dim,
                                                                                          me_dim, ser_dim, s_dim, ac_dim, gc_dim)

        df_atom_name_embedding.to_csv(self.out_atom_name_embed, index=False)

        df_bound_orig_embedding.to_csv(self.out_bound_orig_embed, index=False)

        df_atom_type_embedding.to_csv(self.out_atom_type_embed, index=False)

        df_ab_embedding.to_csv(self.out_bound_ab_embed, index=False)

        df_dl_embedding.to_csv(self.out_bound_dl_embed, index=False)

        df_pf_embedding.to_csv(self.out_carbon_pf_embed, index=False)

        df_mono_accurate_embedding.to_csv(self.out_monosaccharide_accurate_embed, index=False)

        df_mono_simple_embedding.to_csv(self.out_monosaccharide_simple_embed, index=False)

        df_me_embedding.to_csv(self.out_me_embed, index=False)

        df_ser_embedding.to_csv(self.out_ser_embed, index=False)

        df_s_embedding.to_csv(self.out_s_embed, index=False)

        df_ac_embedding.to_csv(self.out_ac_embed, index=False)

        df_gc_embedding.to_csv(self.out_gc_embed, index=False)


def main():
    # C = create_node_embeddings()
    # df_atom_embedding, df_residual_embedding, df_monosaccharide_embedding = C.create_all_embeddings(atom_dim=512, residual_dim=64, mono_dim=64)
    # df_atom_embedding.to_csv(C.out_atom_name_embed, index=False)
    # df_residual_embedding.to_csv(C.out_residual_embed, index=False)
    # df_monosaccharide_embedding.to_csv(C.out_monosaccharide_embed, index=False)
    pass


if __name__ == "__main__":
    main()
