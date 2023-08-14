<p align='center'>
  <img width='50%' src='GlycoNMR_logo.png' />
</p>

# GlycoNMR

Data repository for Carbohydrate-Specific Nuclear Magnetic Resonance Chemical Shift Prediction.

## Datasets:

Curated datasets, click the link to download. 

### 1. Collected and simulated carbohydrate datasets: 

Datasets in format of Protein Data Bank (PDB) with corresponding NMR shift files. Designed for glycoscience researchers. 

- GODESS can be downloaded [here](https://drive.google.com/file/d/15qIixe-irZyJKzvuoINuK1-d53nC8Jyh/view?usp=sharing). <br />
- Glycoscience.DB can be downloaded [here](https://drive.google.com/file/d/1z6OMzvvALq8rOZBWAW5C9yBr_UBQ__ZT/view?usp=sharing). <br />

### 2. Processed carbohydrate datasets:

Processed and annotated tabular datasets. Designed for machine learning researchers. 

- Processed GODESS can be downloaded [here](https://drive.google.com/file/d/1rapUjHs0hhjNfsNMkap3bAdwdNPE2vXA/view?usp=sharing). <br />
- Processed Glycoscience.DB can be downloaded [here](https://drive.google.com/file/d/1z6OMzvvALq8rOZBWAW5C9yBr_UBQ__ZT/view?usp=sharing). <br />

## Usage of GlycoNMR dataset:

### 2D GNN examples

An example of training GCN model on GODESS dataset.

Create graph data for [GODESS](https://drive.google.com/file/d/1rapUjHs0hhjNfsNMkap3bAdwdNPE2vXA/view?usp=sharing)
```python
import torch
from model_2d.NMR_gcn import NMR_GCN
from train_evaluate_2d import NMR_prediction
from preprocess.create_node_embedding.node_embeddings_godess import create_node_embeddings
from preprocess.build_adjaency_matrix.create_adjaency_matrix_godess import build_adjacency_matrix
from preprocess.create_graph.create_graph_data_godess import create_graph

# Initialize node embeddings.
data_dir = 'godess/data/'
C = create_node_embeddings(data_dir)
C.write_all_embeddings()

# Construct edges between atoms.
B = build_adjacency_matrix(data_dir)
B.calculate_all_matrix()

# Construct carbohydrate/glycan graph.
Create = create_graph(data_dir, num_test)
g, test_index = Create.create_all_graph()
```
Train a GCN model for NMR carbon shift, evaluate on the test set.. 

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# atom features
features = g.ndata['z']

# NMR shift values, set to -1 if the shift is not in the main carbon ring. 
labels = g.ndata['y']

# Masked NMR shift values. Indicate training / testing set.
masks = g.ndata['train_carbon_mask'], g.ndata['test_carbon_mask']

# Initialize the model.
model = NMR_GCN().to(device)

NMR_prediction = NMR_prediction(num_epoch = num_epoch, lr = lr)

NMR_prediction.train(g, features, labels, masks, model)

```
