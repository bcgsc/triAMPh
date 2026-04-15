from dataset import triAMPhTestData
from model import Projector, triAMPh
from utils import calc_metrics, load_config

import torch
import numpy as np
import os
import pandas as pd
from loguru import logger
import argparse

def predict_based_on_best_validation(path, prefix, data, graph_idx):
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

        with torch.no_grad():
            processed_p_emb, processed_g_emb = projector(data.amp_embeddings.to(device), data.target_embeddings.to(device))
            pos_scores, neg_scores = model(processed_p_emb, processed_g_emb, data.graphs[graph_idx].to(device), data.graphs[graph_idx+1].to(device), data.graphs[graph_idx+2].to(device), device)
    
        calc_metrics(data.graphs[graph_idx+1].edges(etype='is_active')[1], data.graphs[graph_idx+2].edges(etype='is_active')[1], 
                pos_scores, neg_scores, vacc, vf1, vmf1, vprecision, vrecall, vmcc, vauroc, vavp)
        logger.info(f"Test Accuracy: {vacc[0]}")
        logger.info(f"Test Micro F1: {vf1[0]}")
        logger.info(f"Test Macro F1: {vmf1[0]}")
        logger.info(f"Test Precision: {vprecision[0]}")
        logger.info(f"Test Recall: {vrecall[0]}")
        logger.info(f"Test MCC: {vmcc[0]}")
        logger.info(f"Test AUROC: {vauroc[0]}")
        logger.info(f"Test Average Precision: {vavp[0]}")

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
            "--test_positive_edges",
            help="Path to the file that contains the supervision/test positive edges.\nExpects a .csv file.",
            type=str,
            required=True,
        )
        parser.add_argument(
            "-a",
            "--test_negative_edges",
            help="Path to the file that contains the supervision/test negative edges.\nExpects a .csv file.",
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
            help="Output directory of the triAMPh model of interest. Weights would be located in a subfolder of this directory.",
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

        return parser.parse_args()

def main():
    args = get_args()
    data = triAMPhTestData(args.positive_edges,
                    args.negative_edges,
                    args.test_positive_edges,
                    args.test_negative_edges,
                    args.protein_emb_dir,
                    args.genomic_emb_dir)
    logger.info("Test graphs have been generated according to the following parameters.")
    logger.info(f"Positive message passing file: {args.positive_edges}")
    logger.info(f"Used negative pairs in the training file: {args.positive_edges}")
    logger.info(f"Positive test pairs: {args.test_positive_edges}")
    logger.info(f"Negative test pairs: {args.test_positive_edges}")

    predict_based_on_best_validation(args.output_dir, args.prefix, data, 0)

if __name__ == "__main__":
        main()