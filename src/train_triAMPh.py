from dataset import  triAMPData, triAMPDataInductive
from model import triAMPh, triAMPhDotProd
from utils import set_seed, EarlyStopper, get_timestamp, calc_metrics, save_confusion_matrix, plot_metrics, loss_bce_weighted
import torch
import constants
import argparse
import os
import pandas as pd

def train_validate_test(path, prefix, 
                        graphs, amp_embeddings, target_embeddings,
                        inductive, test, early_stopper, num_epochs, 
                        learning_rate, step_size, gamma,
                        genomic_emb_size, protein_emb_size, han_in_size, han_hidden_size, han_num_heads, han_dropout, relu_after_w, seed):
        
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
        model = triAMPh(meta_paths=meta_paths, 
                genomic_embedding_size=genomic_emb_size,
                protein_embedding_size=protein_emb_size,
                han_in_size=han_in_size,
                han_hidden_size=han_hidden_size,
                han_num_heads=han_num_heads,
                han_dropout=han_dropout,
                relu_after_w=relu_after_w)

        losses = []
        vlosses = []
        tacc = []
        vacc = []
        tf1 = []
        vf1 = []
        tprecision = []
        vprecision = []
        trecall = []
        vrecall = []
        best_val_acc = 0

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        if step_size != 0 and gamma != 0:
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        for i in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                if not inductive:
                        pos_scores, neg_scores = model(amp_embeddings, target_embeddings, graphs[0], graphs[1], graphs[2], device)
                else:
                        pos_scores, neg_scores = model(amp_embeddings[0], target_embeddings, graphs[0], graphs[1], graphs[2], device)

                loss = loss_bce_weighted(pos_scores, neg_scores, device)
                calc_metrics(pos_scores, neg_scores, tacc, tf1, tprecision, trecall)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                if step_size != 0 and gamma != 0:
                        scheduler.step()
                
                model.eval()
                with torch.no_grad():
                        if not inductive:
                                vpos_scores, vneg_scores = model(amp_embeddings, target_embeddings, graphs[3], graphs[4], graphs[5], device)
                        else: 
                                vpos_scores, vneg_scores = model(amp_embeddings[1], target_embeddings, graphs[3], graphs[4], graphs[5], device)
                        calc_metrics(vpos_scores, vneg_scores, vacc, vf1, vprecision, vrecall)
                        loss = loss_bce_weighted(vpos_scores, vneg_scores, device)
                        vlosses.append(loss.item())
                
                # save the model that gives the best accuracy in validation set
                if best_val_acc < vacc[i]:
                        torch.save(model.state_dict(), os.path.join(path, "weights", f"weight_{timestamp}.pth"))
                        bes_val_acc = vacc[i]
                if early_stopper.check(vlosses[i]):
                        break   

                if i % 50 == 0:
                        save_confusion_matrix(path, prefix, i, vpos_scores, vneg_scores)
                
                plot_metrics(path, prefix,
                                losses, vlosses, tacc, vacc, 
                                tf1, vf1, tprecision, vprecision, 
                                trecall, vrecall)
        
        pd.DataFrame({"Loss":losses, "Accuracy":tacc, "F1": tf1, "Precision":tprecision, "Recall":trecall}).to_csv(os.path.join(path, "training_metrics.csv"))
        pd.DataFrame({"Loss":vlosses, "Accuracy":vacc, "F1": tf1, "Precision":vprecision, "Recall":vrecall}).to_csv(os.path.join(path, "validation_metrics.csv"))

        if test:
                acc = []
                f1 = []
                precision = []
                recall = []
                
                model.eval()
                with torch.no_grad():
                        if not inductive:
                                pos_scores, neg_scores = model(amp_embeddings, target_embeddings, graphs[6], graphs[7], graphs[8], device)
                        else:
                                pos_scores, neg_scores = model(amp_embeddings[2], target_embeddings, graphs[6], graphs[7], graphs[8], device)
                        
                        calc_metrics(pos_scores, neg_scores, acc, f1, precision, recall)
                
                save_confusion_matrix(path, prefix+"_test", i, pos_scores, neg_scores)
                print(f"Test- Accuracy: {acc[0]}, F1: {f1[0]}, Precision: {precision[0]}, Recall: {recall[0]}")
        
        return True

def get_args():
        parser = argparse.ArgumentParser()
       
        # Path arguments
        parser.add_argument(
                "-p",
                "--positive_edges",
                help="Path to the file that contains the positive edges.\nExpects a .csv file.",
                type=str,
                required=True,
        )
        parser.add_argument(
                "-n",
                "--negative_edges",
                help="Path to the file that contains the negative edges.\nExpects a .csv file.",
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
                "--prefix",
                help="Prefix to be added to the filenames of the plots and weights generated.",
                type=str,
                required=False,
                default = ""
        )
        
        # Numerical arguments
        parser.add_argument(
                "--tr_split",
                help="Percentage of the training split from the provided data.",
                type=int,
                required=False,
                default = 70
        )
        parser.add_argument(
                "--val_split",
                help="Percentage of the validation split from the provided data.",
                type=int,
                required=False,
                default = 10
        )
        parser.add_argument(
                "--msg_pas",
                help="Percentage of the edges to be used for message passing.",
                type=int,
                required=False,
                default = 80
        )
        parser.add_argument(
                "--inductive",
                help="Training strategy: Inductive if 1, transductive otherwise.",
                type=int,
                required=False,
                default = 0
        )

        parser.add_argument(
                "--lr",
                help="Learning rate for training.",
                type=float,
                required=False,
                default = 0.0001
        )

        parser.add_argument(
                "--epochs",
                help="Number of epochs to train for.",
                type=int,
                required=False,
                default = 750
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
                help="Input length of the projected node vectors given to the Heterogeneous Graph Attention Network.",
                type=int,
                required=False,
                default=256
        )
        parser.add_argument(
                "--han_hidden_size",
                help="Length of the hidden/output node vectors of the Heterogeneous Graph Attention Network.",
                type=int,
                required=False,
                default = 32
        )
        parser.add_argument(
                "--n_heads",
                help="Number of attention heads for Heterogeneous Graph Attention Network.",
                type=int,
                required=False,
                default = 4
        )
        parser.add_argument(
                "--dropout",
                help="Dropout percent for Heterogeneous Graph Attention Network.",
                type=int,
                required=False,
                default = 30
        )
        parser.add_argument(
                "--seed",
                help="Random seed to be set.",
                type=int,
                required=False,
                default = 0
        )
        
        return parser.parse_args()


def main():
        args = get_args()

        os.makedirs(os.path.join(args.output_dir, "accs"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "losses"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "weights"), exist_ok=True)

        test_split = 100 - (args.tr_split + args.val_split)
        if test_split != 0 :
                splits = [args.tr_split/100, args.val_split/100, test_split/100]
        else:
                splits = [args.tr_split/100, args.val_split/100]

        if args.inductive:
                data = triAMPDataInductive(args.positive_edges,
                                args.negative_edges,
                                args.protein_emb_dir,
                                args.genomic_emb_dir,
                                splits, 
                                mes_passing=args.msg_pas/100)

        else:
                data = triAMPData(args.positive_edges,
                                args.negative_edges,
                                args.protein_emb_dir,
                                args.genomic_emb_dir,
                                splits, 
                                mes_passing=args.msg_pas/100)
        stop = 200 
        if stop > args.epochs:
                stop = int(args.epochs/3)

        early_stopper = EarlyStopper(0, stop)
        test = test_split != 0
        train_validate_test(args.output_dir, args.prefix, 
                                data.graphs, data.amp_embeddings, data.target_embeddings,
                                args.inductive, test, early_stopper, args.epochs, 
                                args.lr, 0, 0,
                                args.gen_emb_size, args.prot_emb_size, args.han_input_size, args.han_hidden_size, args.n_heads, args.dropout/100, False,
                                args.seed)


if __name__ == "__main__":
        main()