import os
import random
import datetime

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, matthews_corrcoef, roc_auc_score, average_precision_score
import torch
from matplotlib import pyplot as plt
import pandas as pd

def set_seed(seed = 123):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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


def calc_metrics(pathogen_types_pos, pathogen_types_neg, pos_scores, neg_scores, 
                        acc, f1, macro_f1, precisions, recall, mcc, auroc, avp):
        if torch.cuda.is_available():
                pos = torch.sigmoid(pos_scores).detach().cpu().numpy()
                neg = torch.sigmoid(neg_scores).detach().cpu().numpy()
                pathogen_types_pos = pathogen_types_pos.detach().cpu().numpy()
                pathogen_types_neg = pathogen_types_neg.detach().cpu().numpy()
        else:
                pos = torch.sigmoid(pos_scores).detach().numpy()
                neg = torch.sigmoid(neg_scores).detach().numpy()
                pathogen_types_pos = pathogen_types_pos.detach().cpu().numpy()
                pathogen_types_neg = pathogen_types_neg.detach().cpu().numpy()
        
        scores = np.concatenate((pos, neg))

        pos = (pos >= 0.5).astype(int)
        neg = (neg >= 0.5).astype(int)
        pred = np.concatenate((pos, neg))
        y_pos = np.ones(len(pos))
        y_neg = np.zeros(len(neg))
        y = np.concatenate((y_pos, y_neg))
        pathogens = np.concatenate((pathogen_types_pos, pathogen_types_neg))
        m_f1 = []
        
        for pathogen in set(pathogens):
                indeces = np.where(pathogens == pathogen)
                m_f1.append(f1_score(y[indeces], pred[indeces]))

        acc.append(accuracy_score(y, pred))
        f1.append(f1_score(y, pred))
        macro_f1.append(np.mean(m_f1))
        precisions.append(precision_score(y, pred, zero_division=0))
        recall.append(recall_score(y, pred, zero_division=0))
        mcc.append(matthews_corrcoef(y, pred))
        auroc.append(roc_auc_score(y, scores))
        avp.append(average_precision_score(y, scores))

def plot_metrics(path, prefix, lr, step_size, gamma,
                tloss, vloss, tacc, vacc, 
                tf1, vf1, tmf1, vmf1,
                tprecision, vprecision, 
                trecall, vrecall):
        epochs = np.arange(0, len(tloss))
        plt.plot(epochs, tacc, label = "Training Accucacy")
        plt.plot(epochs, vacc, label = "Validation Accuracy")
        plt.legend()
        plt.title(f"Accuracy")
        plt.xlabel("Epoch Number")
        plt.ylabel("Accuracy")
        plt.savefig(f'{path}/{prefix}_acc.png')
        plt.clf()

        plt.plot(epochs, tloss, label = "Training Loss")
        plt.plot(epochs, vloss, label = "Validation Loss")
        plt.legend()
        plt.title(f"Loss")
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss")
        plt.savefig(f'{path}/{prefix}_loss.png')
        plt.clf()

        plt.plot(epochs, tf1, label = "Training F1")
        plt.plot(epochs, vf1, label = "Validation F1")
        plt.legend()
        plt.title(f"F1")
        plt.xlabel("Epoch Number")
        plt.ylabel("F1 Score")
        plt.savefig(f'{path}/{prefix}_f1.png')
        plt.clf()

        plt.plot(epochs, tmf1, label = "Training Macro-F1")
        plt.plot(epochs, vmf1, label = "Validation Macro-F1")
        plt.legend()
        plt.title(f"Macro-F1")
        plt.xlabel("Epoch Number")
        plt.ylabel("Macro-F1 Score")
        plt.savefig(f'{path}/{prefix}_macro_f1.png')
        plt.clf()

        plt.plot(epochs, trecall, label = "Training Recall")
        plt.plot(epochs, vrecall, label = "Validation Recall")
        plt.legend()
        plt.title(f"Recall")
        plt.xlabel("Epoch Number")
        plt.ylabel("Recall")
        plt.savefig(f'{path}/{prefix}_recall.png')
        plt.clf()

        plt.plot(epochs, tprecision, label = "Training Precision")
        plt.plot(epochs, vprecision, label = "Validation Precision")
        plt.legend()
        plt.title(f"Precision")
        plt.xlabel("Epoch Number")
        plt.ylabel("Precision")
        plt.savefig(f'{path}/{prefix}_precision.png')
        plt.clf()

def loss_bce_weighted(pos_score, neg_score, device):
        # Concatenate predictions and labels
        scores = torch.cat((pos_score, neg_score)).to(device)
        labels = torch.cat((
                torch.ones(pos_score.size(0), device=device),
                torch.zeros(neg_score.size(0), device=device)
        ))

        # Inverse frequency weighting
        pos_weight = neg_score.size(0) / pos_score.size(0)
        weights = torch.cat((
                pos_weight * torch.ones(pos_score.size(0), device=device),
                torch.ones(neg_score.size(0), device=device)
        ))

        # Normalize to stabilize
        weights = weights / weights.mean()

        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels, weight=weights)
        return loss

def loss_bce_unweighted(pos_score, neg_score, device):
        # Concatenate predictions and labels
        scores = torch.cat((pos_score, neg_score)).to(device)
        labels = torch.cat((
                torch.ones(pos_score.size(0), device=device),
                torch.zeros(neg_score.size(0), device=device)
        ))

        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
        return loss

def scan_best_validation(path, prefix, metric):
        val_performance = pd.read_csv(os.path.join(path, f'{prefix}_validation_metrics.csv')).sort_values(by=metric, ascending=False)
        return val_performance.iloc[0]["Epoch"], val_performance.iloc[0][metric]

def load_config(path, prefix, metric='MacroF1'):
        epoch, metric_score=scan_best_validation(path, prefix, metric)

        best_dict = {}
        with open(os.path.join( path, f'{prefix}_log.txt'), 'r') as f:
            for line in f:
                if ':' in line:
                    best_dict[line.split(':')[0]] = line.split(':')[1][:-1]
        best_dict['Epoch']=epoch

        return best_dict

