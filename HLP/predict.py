from dataset import triAMPhTestData
from model import Projector, triAMPh
from utils import calc_metrics, load_config

import torch
import numpy as np
import os
import pandas as pd
from loguru import logger
import argparse
from dgl import heterograph

def predict_based_on_best_validation(path, prefix, data, graph_idx, filename):
        best_dict = load_config(path, prefix)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info("Testing is starting.")
        logger.info(f"Utilized model path: {os.path.join(path, prefix)}")
        logger.info(f"The best validation epoch in terms of Macro F1 score is: {int(best_dict['Epoch'])}")

        projector = Projector(proteomic_emb_size=1280, genomic_emb_size=512, han_in_size=int(best_dict['HAN input size'])).to(device)
        model = triAMPh(meta_paths = [["is_active", "is_similar_g"],["is_susceptable", "is_similar_p"], ["is_similar_p", "is_active"], ["is_similar_g", "is_susceptable"], ["is_active", "is_susceptable"], ["is_susceptable", "is_active"]], 
                        han_in_size=int(best_dict['HAN input size']),
                        han_hidden_size=int(best_dict['HAN Hidden size']),
                        han_num_heads=int(best_dict['Number of attention heads for GAT']),
                        han_dropout=float(best_dict['HAN Dropout']),
                        sem_att_size=int(best_dict['Semantic Attention Vector Size']),
                        p_percentile=float(best_dict['Peptide Distance Percentile Cutoff']),
                        g_percentile=float(best_dict['Genome Distance Percentile Cutoff']),
                        add_self_embeddings=best_dict['Add Self Embeddings']=="True",
                        add_self_loops=best_dict['Add Self Loops']=="True").to(device)
        

        projector.load_state_dict(torch.load(os.path.join(best_dict['Path'], 'weights', f'{prefix}_projector_weight_{int(best_dict["Epoch"])}.pth'), weights_only=False, map_location=device))
        model.load_state_dict(torch.load(os.path.join(best_dict['Path'], 'weights', f'{prefix}_triAMPh_weight_{int(best_dict["Epoch"])}.pth'), weights_only=False, map_location=device))

        model.eval()
        projector.eval()
        vacc, vf1, vmf1, vprecision, vrecall, vmcc, vauroc, vavp= [], [], [], [], [], [], [], []

        dummy_graph = heterograph({
                    ("AMP", "is_similar_p", "AMP"): ([0], [0]),
                    ("Target", "is_similar_g", "Target"): ([0], [0]),
                    ("AMP", "is_active", "Target"): ([0], [0]),
                    ("Target", "is_susceptable", "AMP"): ([0], [0])
        }, num_nodes_dict = {'AMP': data.amp_embeddings.size()[0], 'Target': data.target_embeddings.size()[0]})

        with torch.no_grad():
            processed_p_emb, processed_g_emb = projector(data.amp_embeddings.to(device), data.target_embeddings.to(device))
            pos_scores, neg_scores = model(processed_p_emb, processed_g_emb, data.graphs[graph_idx].to(device), data.graphs[graph_idx+1].to(device), dummy_graph, device)

        if torch.cuda.is_available():
            pos = torch.sigmoid(pos_scores).detach().cpu().numpy()
            neg = torch.sigmoid(neg_scores).detach().cpu().numpy()
        else:
            pos = torch.sigmoid(pos_scores).detach().numpy()
            neg = torch.sigmoid(neg_scores).detach().numpy()
        pos = (pos >= 0.5).astype(int)
        neg = (neg >= 0.5).astype(int)
        
        results = pd.DataFrame({'AMP_ID':data.graphs[graph_idx+1].edges(etype='is_active')[0], 'Target_ID':data.graphs[graph_idx+1].edges(etype='is_active')[1], 'Prediction': pos, 'Prediction Score':torch.sigmoid(pos_scores)})
        amp_idx = list(data.amp_idx.values())
        amps = list(data.amp_idx.keys())
        target_idx = list(data.target_idx.values())
        targets = list(data.target_idx.keys())
        ider = pd.DataFrame({'ID':amps, 'AMP_ID':map(int, amp_idx)})
        results = pd.merge(results, ider, how='left')
        ider = pd.DataFrame({ 'Pathogens':targets, 'Target_ID':map(int, target_idx)})
        results = pd.merge(results, ider, how='left')
        results = results[['ID' , 'Pathogens', 'Prediction', 'Prediction Score']]
        results.to_csv(os.path.join(path, prefix, filename))
        logger.info(f"The predictions were saved to {os.path.join(path, filename)}")

        return True

def get_args():
        parser = argparse.ArgumentParser()
       
        # Path arguments
        parser.add_argument(
            "-p",
            "--positive_edges",
            help="Path to the file that contains the message passing positive edges.\nExpects a .csv file.",
            type=str,
            required=True,
        )
        parser.add_argument(
            "-n",
            "--negative_edges",
            help="Path to the file that contains the message passing negative edges.\nExpects a .csv file.",
            type=str,
            required=True,
        )
        parser.add_argument(
            "-t",
            "--query_edges",
            help="Path to the file that contains the edges to be predicted.\nExpects a .csv file.",
            type=str,
            required=True,
        )

        parser.add_argument(
            "-e",
            "--protein_emb_dir",
            help="Path to the folder that contains the individual embeddings of peptides.\n Note: Files should be saved in .npy format.",
            type=str,
            required=True,
        )
        parser.add_argument(
            "-g",
            "--genomic_emb_dir",
            help="Path to the folder that contains the individual embeddings of pathogens.\n Note: Files should be saved in .npy format.",
            type=str,
            required=True,
        )

        parser.add_argument(
            "-o",
            "--output_dir",
            help="Output directory of the triAMPh model of interest. Weights would be located in a subfolder of this directory. This will also be the folder the results will be written to.",
            type=str,
            required=True,
        )

        parser.add_argument(
            "--prefix",
            help="Prefix added to the filenames of the plots and weights of interest.",
            type=str,
            required=False,
            default = ""
        )

        parser.add_argument(
            "--filename",
            help="Name of the file that the results will be written to. Append .csv to the end.",
            type=str,
            required=False,
            default = "triAMPh_results.csv"
        )

        return parser.parse_args()

def main():
    args = get_args()
    data = triAMPhTestData(args.positive_edges,
                    args.negative_edges,
                    args.query_edges,
                    args.query_edges,
                    args.protein_emb_dir,
                    args.genomic_emb_dir)
    logger.info("Test graphs have been generated according to the following parameters.")
    logger.info(f"Positive message passing file: {args.positive_edges}")
    logger.info(f"Used negative pairs in the training file: {args.positive_edges}")
    logger.info(f"Query/Test pairs: {args.query_edges}")
    
    predict_based_on_best_validation(args.output_dir, args.prefix, data, 0, args.filename)

if __name__ == "__main__":
        main()