from dataset import triAMPhDataTest
import constants
import utils
import pandas as pd
import numpy as np
import torch
from model import triAMPh
import argparse

def test_triAMPh(   data, path, weight_path, 
                    threshold, 
                    genomic_emb_size, protein_emb_size, han_in_size, han_hidden_size, han_num_heads, relu_after_w,
                    seed):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: ", device)
        if seed == 0:
                set_seed(constants.SEED)
                print(f'Seed set: {constants.SEED}')
        else:
                set_seed(seed)
                print(f'Seed set: {seed}')
        
        timestamp = get_timestamp()
        set_seed(constants.SEED)
        print(f"Training started: {timestamp}")
        meta_paths = [["is_active", "is_susceptable"],["is_susceptable", "is_active"], ["is_similar_g", "is_susceptable"], ["is_similar_p", "is_active"]]
        model = triAMP(meta_paths=meta_paths, 
                genomic_embedding_size=genomic_emb_size,
                protein_embedding_size=protein_emb_size,
                han_in_size=han_in_size,
                han_hidden_size=han_hidden_size,
                han_num_heads=han_num_heads,
                han_dropout=0,
                relu_after_w=relu_after_w)

        model.load_state_dict(torch.load(weight_path))

        # test
        torch.cuda.empty_cache()
        acc = []
        f1 = []
        precision = []
        recall = []
        
        model.eval()
        with torch.no_grad():
                pos_scores, neg_scores = model(data.amp_embeddings, data.target_embeddings, data.graphs[0], data.graphs[1], data.graphs[2], device)
                utils.calc_metrics(pos_scores, neg_scores, acc, f1, precision, recall, threshold=threshold)
        utils.save_confusion_matrix(path, prefix, 0, pos_scores, neg_scores)
        print(f"Test - Accuracy: {acc[0]}, F1: {f1[0]}, Precision: {precision[0]}, Recall: {recall[0]}")
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
                help="Path to the file that contains the message passing positive edges.\nExpects a .csv file.",
                type=str,
                required=True,
        )
        parser.add_argument(
                "-a",
                "--test_negative_edges",
                help="Path to the file that contains the message passing negative edges.\nExpects a .csv file.",
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
                help="Path to the directory where the results will be saved at",
                type=str,
                required=True,
        )

        parser.add_argument(
                "-w",
                "--weight_path",
                help="Path to the pretrained weights of triAMPh.",
                type=str,
                required=True,
        )

        # Numerical arguments
        parser.add_argument(
                "--threshold",
                help="Threshold value for binary cross entropy.\nDefault: Above 0.5 positive, below negative.",
                type=int,
                required=False,
                default = 0.5
        )
        parser.add_argument(
                "--gen_emb_size",
                help="Length of the genomic embedding vector.",
                type=int,
                required=True
        )
        parser.add_argument(
                "--prot_emb_size",
                help="Length of the protein embedding vector.",
                type=int,
                required=True
        )
        parser.add_argument(
                "--han_input_size",
                help="Input length of the projected node vectors given to the Heterogeneous Graph Attention Network",
                type=int,
                required=False,
                default=256
        )
        parser.add_argument(
                "--han_hidden_size",
                help="Length of the hidden/output node vectors of the Heterogeneous Graph Attention Network",
                type=int,
                required=False,
                default = 32
        )
        parser.add_argument(
                "--n_heads",
                help="Number of attention heads for Heterogeneous Graph Attention Network",
                type=int,
                required=False,
                default = 4
        )
        
        parser.add_argument(
                "--seed",
                help="Dropout percent for Heterogeneous Graph Attention Network",
                type=int,
                required=False,
                default = 0
        )
        
        return parser.parse_args()



def main():
    args = get_args()
    data = triAMPhDataTest(args.positive_edges,
                    args.negative_edges,
                    args.test_positive_edges,
                    args.test_negative_edges,
                    args.protein_emb_dir,
                    args.genomic_emb_dir)

    test_triAMPh(data, args.output_dir, args.weight_path, 
                        args.threshold, 
                        args.gen_emb_size, args.prot_emb_size, args.han_input_size, args.han_hidden_size, args.han_num_heads, False,
                        args.seed)

if __name__ == "__main__":
        main()