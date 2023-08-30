# @Author  : Zizhang Chen
# @Contact : zizhang2@outlook.com
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from model_2d.NMR_gcn import NMR_GCN
from preprocess.create_graph.create_graph_data_godess import create_graph
from tqdm import tqdm
from sklearn.metrics import r2_score

class NMR_prediction:
    def __init__(self, results_dir='dataset/Godess_final_data/results/training_hydrogen.csv',
                 results_dir_test = 'dataset/Godess_final_data/results/testing_hydrogen.csv',
                 model_dir='model_state/Model_no_residual_embed_Carbon_best_only_node_embedding.pt',
                 num_epoch=1000, lr=1e-2, weight_decay=5e-4):
        self.results_dir = results_dir
        self.results_dir_test = results_dir_test
        self.model_dir = model_dir
        self.num_epoch = num_epoch
        self.lr = lr
        self.weight_decay = weight_decay

    def evaluate(self, g, features, shift_values, mask, model, save_train=False, save_test = False, report_r2 = False):
        model.eval()
        with torch.no_grad():
            predict_shift = model(g, features)
            predict_shift_test = predict_shift[mask].cpu().numpy()
            actual_shift_test = shift_values[mask].cpu().numpy()

            correct = np.sum((predict_shift_test - actual_shift_test) ** 2)
            r_2 = r2_score(predict_shift_test, actual_shift_test)


            if save_train:
                df_temp = pd.DataFrame([predict_shift_test, actual_shift_test]).T
                df_temp.to_csv(self.results_dir, index=False)
            # print(len(predict_shift_test))

            if save_test:
                df_temp = pd.DataFrame([predict_shift_test, actual_shift_test]).T
                df_temp.to_csv(self.results_dir_test, index=False)

            # return np.sqrt(correct.item() * 1.0 / len(predict_shift_test))

            if report_r2:
                return np.sqrt(correct / len(predict_shift_test)), r_2

            else:
                return np.sqrt(correct / len(predict_shift_test))

    def train(self, g, features, shift_values, masks, model, report_r2 = False):
        # define train/val samples, loss function and optimizer
        train_mask = masks[0]
        test_mask = masks[1]
        loss_fcn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss = 100000
        # training loop
        for epoch in tqdm(range(self.num_epoch)):
            model.train()
            logits = model(g, features)
            loss = loss_fcn(logits[train_mask], shift_values[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if report_r2:
                mse_test, r2_test = self.evaluate(g, features, shift_values, test_mask, model, report_r2 = report_r2)
                mse_train, r2_train = self.evaluate(g, features, shift_values, train_mask, model, report_r2 = report_r2)
                print(
                    "Epoch {:05d} | Loss {:.4f} | train_RMSE {:.4f} | train_r2 {:.4f} | test_RMSE {:.4f} | test_r2 {:.4f} ".format(
                        epoch, loss.item(), mse_train, r2_train, mse_test, r2_test
                    )
                )
            else:
                mse_test = self.evaluate(g, features, shift_values, test_mask, model)
                mse_train = self.evaluate(g, features, shift_values, train_mask, model)
                print(
                    "Epoch {:05d} | Loss {:.4f} | train_RMSE {:.4f} | test_RMSE {:.4f} ".format(
                        epoch, loss.item(), mse_train, mse_test
                    )
                )
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), self.model_dir)

            if epoch % 1000 == 0:
                self.lr = self.lr * 0.8
        print('best loss:', best_loss)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    C = create_graph()
    g, test_index = C.create_all_graph()

    pd.DataFrame(test_index).to_csv('data/test_index.csv', index=False)

    g = g.int()
    g = g.to(device)
    features = g.ndata["feat"]
    labels = g.ndata["shift_value"]
    # masks = g.ndata['train_mask'], g.ndata['test_mask']
    # masks = g.ndata['train_hydrogen_mask'], g.ndata['test_hydrogen_mask']
    masks = g.ndata['train_carbon_mask'], g.ndata['test_carbon_mask']
    print(features.dtype)
    print(labels.dtype)
    # model = NMR_GCN(in_size=576, hid_size=[256, 128, 64, 32], out_size=1).to(device)
    model = NMR_GCN(in_size=512, hid_size=[256, 128, 64, 32], out_size=1).to(device)
    # model training

    NMR_prediction = NMR_prediction()
    print("Training...")
    NMR_prediction.train(g, features, labels, masks, model)

    # test the model
    print("Testing...")
    saved_model = NMR_GCN(in_size=512, hid_size=[256, 128, 64, 32], out_size=1).to(device)
    saved_model.load_state_dict(torch.load(NMR_prediction.model_dir))
    acc = NMR_prediction.evaluate(g, features, labels, masks[1], saved_model)
    print("MSE {:.4f}".format(acc))
