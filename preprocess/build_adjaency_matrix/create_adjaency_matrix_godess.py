# @Author  : Zizhang Chen
# @Contact : zizhang2@outlook.com
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


class build_adjacency_matrix:
    def __init__(self, labeled_pdb_dir='godess/data',
                 out_adjacency_dir='godess/adjaency_matrix/',
                 out_interaction_dir='godess/adjaency_matrix/', threshold_carbon=1.65,
                 threshold_hydrogen=1.18,
                 threshold_general=1.5, threshold_interaction=5.0,
                 carbon_list=None,
                 hydrogen_list=None):
        if carbon_list is None:
            # carbon_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
            carbon_list = ['C']
        if hydrogen_list is None:
            # hydrogen_list = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9']
            hydrogen_list = ['H']
        self.labeled_pdb_dir = labeled_pdb_dir
        self.out_adjacency_dir = out_adjacency_dir
        self.out_interaction_dir = out_interaction_dir
        self.threshold_carbon = threshold_carbon
        self.threshold_hydrogen = threshold_hydrogen
        self.threshold_general = threshold_general
        self.threshold_interaction = threshold_interaction
        self.carbon_list = carbon_list
        self.hydrogen_list = hydrogen_list

    def calculate_single_matrix(self, file_name):
        df1 = pd.read_csv(self.labeled_pdb_dir + file_name)
        df1['Atom'] = df1.index
        temp_atom_type = df1['Atom_type'].values
        coordinate_matrix = np.array(df1.loc[:, ['x', 'y', 'z']])
        distance_matrix = np.zeros([len(df1), len(df1)])
        adjacency_matrix = np.zeros([len(df1), len(df1)])
        adjacency_matrix_interaction = np.zeros([len(df1), len(df1)])
        """
        Calculate the distance matrix
        """
        for i in range(len(df1)):
            distance_matrix[i:] = np.sqrt(np.sum((coordinate_matrix[i] - coordinate_matrix) ** 2, axis=1))

        """
        Assign edges and interaction by distance matrix
        """

        for i in range(len(df1)):
            for j in range(len(df1)):
                # build up edges by threshold
                if (temp_atom_type[i] in self.carbon_list) or (temp_atom_type[j] in self.carbon_list):
                    if distance_matrix[i][j] < self.threshold_carbon:
                        adjacency_matrix[i][j] = 1

                elif (temp_atom_type[i] in self.hydrogen_list) or (temp_atom_type[j] in self.hydrogen_list):
                    if distance_matrix[i][j] < self.threshold_hydrogen:
                        adjacency_matrix[i][j] = 1

                else:
                    if distance_matrix[i][j] < self.threshold_general:
                        adjacency_matrix[i][j] = 1
                # build up interaction edge by threshold
                if distance_matrix[i][j] < self.threshold_general:
                    adjacency_matrix_interaction[i][j] = 1

        return adjacency_matrix, adjacency_matrix_interaction

    def calculate_all_matrix(self):
        all_files = os.listdir(self.labeled_pdb_dir)
        for f in tqdm(all_files):
            adjacency_matrix, adjacency_matrix_interaction = self.calculate_single_matrix(f)
            adjacency_matrix_name = self.out_adjacency_dir + 'edges_' + f
            adjacency_matrix_interaction_name = self.out_interaction_dir + 'interaction_' + f

            adjacency_matrix = pd.DataFrame(adjacency_matrix)
            adjacency_matrix.to_csv(adjacency_matrix_name, index=False)

            adjacency_matrix_interaction = pd.DataFrame(adjacency_matrix_interaction)
            adjacency_matrix_interaction.to_csv(adjacency_matrix_interaction_name, index=False)


def main():
    C = build_adjacency_matrix()
    C.calculate_all_matrix()


if __name__ == "__main__":
    main()
