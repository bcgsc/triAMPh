from Bio import Entrez
from Bio.SeqIO.FastaIO import SimpleFastaParser, FastaWriter
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

import os
import random
import numpy as np
from torch import manual_seed, use_deterministic_algorithms
import datetime

from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn import BCELoss
import torch

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx
import dgl 

def set_seed(seed = 123):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    manual_seed(seed)

def get_timestamp():
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp
    
class EarlyStopper:
    def __init__(self, epsilon, 
                 patience):
        self.epsilon = epsilon
        self.patience = patience
        self.counter = 0
        self.min_loss = float("inf")
    
    def check(self, current_loss):
        if current_loss + self.epsilon < self.min_loss:
            self.min_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def calc_metrics(pos_scores, neg_scores, acc, f1, precisions, recall, threshold=.5):
        if torch.cuda.is_available():
                pos = pos_scores.detach().cpu().numpy()
                neg = neg_scores.detach().cpu().numpy()
        else:
                pos = pos_scores.detach().numpy()
                neg = neg_scores.detach().numpy()
        pos = (pos >= 0.5).astype(int)
        neg = (neg >= 0.5).astype(int)
        pred = np.concatenate((pos, neg))
        y_pos = np.ones(len(pos))
        y_neg = np.zeros(len(neg))
        y = np.concatenate((y_pos, y_neg))
        acc.append(accuracy_score(y, pred))
        f1.append(f1_score(y, pred))
        precisions.append(precision_score(y, pred,zero_division=0))
        recall.append(recall_score(y, pred, zero_division=0))

def save_confusion_matrix(path, prefix, i ,pos_scores, neg_scores):
        if torch.cuda.is_available():
                pos = pos_scores.detach().cpu().numpy()
                neg = neg_scores.detach().cpu().numpy()
        else:
                pos = pos_scores.detach().numpy()
                neg = neg_scores.detach().numpy()
        pos = (pos >= 0.5).astype(int)
        neg = (neg >= 0.5).astype(int)
        pred = np.concatenate((pos, neg))
        y_pos = np.ones(len(pos))
        y_neg = np.zeros(len(neg))
        y = np.concatenate((y_pos, y_neg))

        cm = confusion_matrix(y, pred)
        ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Negative Edge', 'Positive Edge']).plot()
        plt.savefig(os.path.join(path,f"{prefix}_epoch_{i}_cm.png"))
        plt.clf()

def plot_metrics(path, prefix,
                tloss, vloss, tacc, vacc, 
                tf1, vf1, tprecision, vprecision, 
                trecall, vrecall):
        epochs = np.arange(0, len(tloss))
        plt.plot(epochs, tacc, label = "Training Accucacy")
        plt.plot(epochs, vacc, label = "Validation Accuracy")
        plt.legend()
        plt.title(f"Accuracy")
        plt.xlabel("Epoch Number")
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(path, "accs", f"{prefix}_accs.png"))
        plt.clf()

        plt.plot(epochs, tloss, label = "Training Loss")
        plt.plot(epochs, vloss, label = "Validation Loss")
        plt.legend()
        plt.title(f"Loss")
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(path, "losses", f"{prefix}_losses.png"))
        plt.clf()

        plt.plot(epochs, tf1, label = "Training F1")
        plt.plot(epochs, vf1, label = "Validation F1")
        plt.legend()
        plt.title(f"F1")
        plt.xlabel("Epoch Number")
        plt.ylabel("F1 Score")
        plt.savefig(os.path.join(path, f"{prefix}_f1s.png"))
        plt.clf()

        plt.plot(epochs, trecall, label = "Training Recall")
        plt.plot(epochs, vrecall, label = "Validation Recall")
        plt.legend()
        plt.title(f"Recall")
        plt.xlabel("Epoch Number")
        plt.ylabel("Recall")
        plt.savefig(os.path.join(path, f"{prefix}_recalls.png"))
        plt.clf()

        plt.plot(epochs, tprecision, label = "Training Precision")
        plt.plot(epochs, vprecision, label = "Validation Precision")
        plt.legend()
        plt.title(f"Precision")
        plt.xlabel("Epoch Number")
        plt.ylabel("Precision")
        plt.savefig(os.path.join(path, f"{prefix}_precisions.png"))
        plt.clf()

def loss_bce(pos_score, neg_score, device):
        pos_score = torch.squeeze(pos_score)
        neg_score = torch.squeeze(neg_score)
        scores = torch.cat((pos_score, neg_score)).to(device)
        labels = torch.cat((torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0]))).to(device)
        loss = BCELoss()
        return loss(scores, labels)

def loss_bce_weighted(pos_score, neg_score, device):
        pos_score = torch.squeeze(pos_score)
        neg_score = torch.squeeze(neg_score)
        scores = torch.cat((pos_score, neg_score)).to(device)
        labels = torch.cat((torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0]))).to(device)
        weight = pos_score.shape[0]
        weight = weight / neg_score.shape[0]
        weights = torch.cat((torch.ones(pos_score.shape[0]), weight * torch.ones(neg_score.shape[0]))).to(device)
        loss = BCELoss(weight=weights)
        return loss(scores, labels)


# adapted from https://www.dgl.ai/dgl_docs/guide/training-link.html
def loss_m(pos_score, neg_score, margin):
        # Margin loss
        n_edges = pos_score.shape[0]
        return (margin - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

# adapted from https://stackoverflow.com/questions/72847822/how-to-visualize-heterodata-pytorch-geometric-graph-with-any-tool
def plot_graph(path, prefix, lr, step_size, gamma, i, g):
        graph = dgl.to_networkx(g.cpu())

        # Define colors for nodes and edges
        node_type_colors = {
        "AMP": "#10F4FF",
        "Target": "#7433ED",
        }
        node_colors = []
        labels = {}
        amps = []
        targets = []
        for node, attrs in graph.nodes(data=True):
                node_type = attrs["ntype"]
                color = node_type_colors[node_type]
                node_colors.append(color)
                if attrs["ntype"] == "AMP":
                        labels[node] = f"AMP{node}"
                        amps.append(node)
                elif attrs["ntype"] == "Target":
                        labels[node] = f"Target{node}"
                        targets.append(node)
        
        # Define colors for the edges
        edge_type_colors = {
                ("AMP", "is_similar_p", "AMP"): "#10F4FF",
                ("Target", "is_similar_g", "Target"): "#7433ED",
                ("Target", "is_susceptable", "AMP"): "#EDB533",
                ("AMP", "is_active", "Target"): "#EDB533",
        }

        edge_colors = []

        for from_node, to_node, attrs in graph.edges(data=True):
                edge_type = attrs["etype"]
                color = edge_type_colors[edge_type]
                graph.edges[from_node, to_node, 0]["color"] = color
                edge_colors.append(color)

        #pos = nx.spring_layout(graph, k=2)
        pos = nx.bipartite_layout(graph, targets)
        nx.draw_networkx(
                graph,
                pos=pos,
                labels=labels,
                with_labels=True,
                node_color=node_colors,
                edge_color=edge_colors,
                node_size=30,
                font_size=2
        )
        plt.savefig(f'{path}/networks/{prefix}_lr_{lr}_step_{step_size}_gamma_{gamma}__epoch_{i}.png')
        plt.clf()
