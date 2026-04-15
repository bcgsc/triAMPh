from dataset import triAMPhTransductiveData, triAMPhInductiveData
from model import Projector, triAMPh
from utils import set_seed, EarlyStopper, get_timestamp, calc_metrics, plot_metrics, loss_bce_weighted, loss_bce_unweighted, load_config

import torch
import numpy as np
import os
import pandas as pd
from loguru import logger
import argparse

def train_inductive(meta_paths, han_in_size, han_hidden_size, mheads, dropout, sem_att_size, p_percentile, g_percentile,
                learning_rate, scheduler_step_size, scheduler_gamma, early_stopper_epsilon, early_stopper_patience, num_epochs, 
                path, prefix,  
                tr_mes_pas, learning_scheme, graphs, amp_embeddings, target_embeddings,
                add_self_embeddings, add_self_loops, prot_emb_size, gen_emb_size, seed):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Using device: ", device)
        set_seed(seed)
        
        amp_embeddings[0] = amp_embeddings[0].to(device)
        amp_embeddings[1] = amp_embeddings[1].to(device)
        target_embeddings = target_embeddings.to(device)
        
        for i in range(6):
                graphs[i] = graphs[i].to(device)
        
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, "weights"), exist_ok=True)

        logger.info(f"Writing the parameters to file: {os.path.join(path,f'{prefix}_log.txt')}")
        with open(os.path.join(path,f"{prefix}_log.txt"), "w") as f:
                f.write(f"Meta paths:{meta_paths}\n")
                f.write(f"HAN input size:{han_in_size}\n")
                f.write(f"HAN Hidden size:{han_hidden_size}\n")
                f.write(f"Number of attention heads for GAT:{mheads}\n")
                f.write(f"HAN Dropout:{dropout}\n")
                f.write(f"Semantic Attention Vector Size:{sem_att_size}\n")
                f.write(f"Peptide Distance Percentile Cutoff:{p_percentile}\n")
                f.write(f"Genome Distance Percentile Cutoff:{g_percentile}\n\n")
                
                f.write(f"Learning Rate:{learning_rate}\n")
                f.write(f"Scheduler Step Size:{scheduler_step_size}\n")
                f.write(f"Scheduler Gamma:{scheduler_gamma}\n")
                f.write(f"Early Stopper Epsilon:{early_stopper_epsilon}\n")
                f.write(f"Early Stopper Patience:{early_stopper_patience}\n")
                f.write(f"Number of Epochs:{num_epochs}\n\n")

                f.write(f"Path:{path}\n")
                f.write(f"Prefix:{prefix}\n")
                f.write(f"Message Passing Proportion in Training:{tr_mes_pas}\n")
                f.write(f"Learning Scheme:{learning_scheme}\n\n")        

                f.write(f"Add Self Embeddings:{add_self_embeddings}\n")
                f.write(f"Add Self Loops:{add_self_loops}\n\n")
        
        early_stopper = EarlyStopper(early_stopper_epsilon, early_stopper_patience)
        projector = Projector(proteomic_emb_size=prot_emb_size, genomic_emb_size=gen_emb_size, han_in_size=han_in_size).to(device)
        model = triAMPh(meta_paths=meta_paths, 
                han_in_size=han_in_size, # bilstm hidden size = han_in_size/2
                han_hidden_size=han_hidden_size,
                han_num_heads=mheads,
                han_dropout=dropout,
                sem_att_size=sem_att_size,
                p_percentile=p_percentile,
                g_percentile=g_percentile,
                add_self_embeddings=add_self_embeddings, 
                add_self_loops=add_self_loops).to(device)

        losses = []
        vlosses = []
        tacc = []
        vacc = []
        tf1 = []
        vf1 = []
        tmf1 = []
        vmf1 = []
        tprecision = []
        vprecision = []
        trecall = []
        vrecall = []
        tmcc = []
        vmcc = []
        tauroc = []
        vauroc = []
        tavp = []
        vavp = []

        optimizer = torch.optim.Adam(list(projector.parameters())+list(model.parameters()), lr=learning_rate)
        if scheduler_step_size != "":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        logger.info("Training is starting.")
        logger.info(f'Training metrics are being written to: {os.path.join(path, prefix)}_train_metrics.csv')
        logger.info(f'Validation metrics are being written to:  {os.path.join(path, prefix)}_validation_metrics.csv')
        logger.info(f'Weights are being written to: {os.path.join(path, "weights")}')
        for i in range(num_epochs):
                logger.info(f"Epoch {i}")
                projector.train()
                model.train()
                optimizer.zero_grad()
                processed_p_emb, processed_g_emb = projector(amp_embeddings[0], target_embeddings)
                pos_scores, neg_scores = model(processed_p_emb, processed_g_emb, graphs[0], graphs[1], graphs[2], device)

                loss = loss_bce_weighted(pos_scores, neg_scores, device)
                calc_metrics(graphs[1].edges(etype='is_active')[1], graphs[2].edges(etype='is_active')[1], 
                                                        pos_scores, neg_scores, tacc, tf1, tmf1, tprecision, trecall, tmcc, tauroc, tavp)
                losses.append(loss.item())
                
                loss.backward()
                optimizer.step()
                if scheduler_step_size != "":
                        scheduler.step()
                model.eval()
                projector.eval()
                with torch.no_grad():
                        processed_p_emb, processed_g_emb = projector(amp_embeddings[1], target_embeddings)
                        vpos_scores, vneg_scores = model(processed_p_emb, processed_g_emb, graphs[3], graphs[4], graphs[5], device)
                        calc_metrics(graphs[4].edges(etype='is_active')[1], graphs[5].edges(etype='is_active')[1], 
                                                        vpos_scores, vneg_scores, vacc, vf1, vmf1, vprecision, vrecall, vmcc, vauroc, vavp)
                        loss = loss_bce_unweighted(vpos_scores, vneg_scores, device)
                        vlosses.append(loss.item())

                if early_stopper.check(vlosses[i]):
                        break   
                
                plot_metrics(path, prefix, learning_rate, scheduler_step_size, scheduler_gamma,
                        losses, vlosses, tacc, vacc, 
                        tf1, vf1, tmf1, vmf1, tprecision, vprecision, 
                        trecall, vrecall)
                
                training_df = pd.DataFrame({"Epoch":np.arange(len(tacc)), "Accuracy":tacc, "Precision":tprecision, "Recall": trecall, "F1": tf1, 'MacroF1':tmf1, "MCC": tmcc, "AUROC": tauroc, "AVP": tavp})
                training_df.to_csv(os.path.join(path, f"{prefix}_train_metrics.csv"))
                validation_df = pd.DataFrame({"Epoch":np.arange(len(vacc)), "Accuracy":vacc, "Precision":vprecision, "Recall": vrecall, "F1": vf1, 'MacroF1':vmf1, "MCC": vmcc, "AUROC": vauroc, "AVP": vavp})
                validation_df.to_csv(os.path.join(path, f"{prefix}_validation_metrics.csv"))

                torch.save(model.state_dict(), os.path.join(path, "weights", f"{prefix}_triAMPh_weight_{i}.pth"))
                torch.save(projector.state_dict(), os.path.join(path, "weights", f"{prefix}_projector_weight_{i}.pth"))
                
                torch.cuda.empty_cache()
        
        return True

def train_transductive(meta_paths, han_in_size, han_hidden_size, mheads, dropout, sem_att_size, p_percentile, g_percentile,
                learning_rate, scheduler_step_size, scheduler_gamma, early_stopper_epsilon, early_stopper_patience, num_epochs, 
                path, prefix,  
                tr_mes_pas, learning_scheme, graphs, amp_embeddings, target_embeddings,
                add_self_embeddings, add_self_loops, prot_emb_size, gen_emb_size, seed):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Using device: ", device)
        set_seed(seed)
        
        amp_embeddings = amp_embeddings.to(device)
        target_embeddings = target_embeddings.to(device)
        
        for i in range(6):
                graphs[i] = graphs[i].to(device)
        
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, "weights"), exist_ok=True)

        logger.info(f"Writing the parameters to file: {os.path.join(path,f'{prefix}_log.txt')}")
        with open(os.path.join(path,f"{prefix}_log.txt"), "w") as f:
                f.write(f"Meta paths:{meta_paths}\n")
                f.write(f"HAN input size:{han_in_size}\n")
                f.write(f"HAN Hidden size:{han_hidden_size}\n")
                f.write(f"Number of attention heads for GAT:{mheads}\n")
                f.write(f"HAN Dropout:{dropout}\n")
                f.write(f"Semantic Attention Vector Size:{sem_att_size}\n")
                f.write(f"Peptide Distance Percentile Cutoff:{p_percentile}\n")
                f.write(f"Genome Distance Percentile Cutoff:{g_percentile}\n\n")
                
                f.write(f"Learning Rate:{learning_rate}\n")
                f.write(f"Scheduler Step Size:{scheduler_step_size}\n")
                f.write(f"Scheduler Gamma:{scheduler_gamma}\n")
                f.write(f"Early Stopper Epsilon:{early_stopper_epsilon}\n")
                f.write(f"Early Stopper Patience:{early_stopper_patience}\n")
                f.write(f"Number of Epochs:{num_epochs}\n\n")

                f.write(f"Path:{path}\n")
                f.write(f"Prefix:{prefix}\n")
                f.write(f"Message Passing Proportion in Training:{tr_mes_pas}\n")
                f.write(f"Learning Scheme:{learning_scheme}\n\n")        

                f.write(f"Add Self Embeddings:{add_self_embeddings}\n")
                f.write(f"Add Self Loops:{add_self_loops}\n\n")

        early_stopper = EarlyStopper(early_stopper_epsilon, early_stopper_patience)

        projector = Projector(proteomic_emb_size=prot_emb_size, genomic_emb_size=gen_emb_size, han_in_size=han_in_size).to(device)
        model = triAMPh(meta_paths=meta_paths, 
                han_in_size=han_in_size, # bilstm hidden size = han_in_size/2
                han_hidden_size=han_hidden_size,
                han_num_heads=mheads,
                han_dropout=dropout,
                sem_att_size=sem_att_size,
                p_percentile=p_percentile,
                g_percentile=g_percentile,
                add_self_embeddings=add_self_embeddings, 
                add_self_loops=add_self_loops).to(device)

        losses = []
        vlosses = []
        tacc = []
        vacc = []
        tf1 = []
        vf1 = []
        tmf1 = []
        vmf1 = []
        tprecision = []
        vprecision = []
        trecall = []
        vrecall = []
        tmcc = []
        vmcc = []
        tauroc = []
        vauroc = []
        tavp = []
        vavp = []

        optimizer = torch.optim.Adam(list(projector.parameters())+list(model.parameters()), lr=learning_rate)
        if scheduler_step_size != "":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        logger.info("Training is starting.")
        logger.info(f'Training metrics are being written to: {os.path.join(path, prefix)}_train_metrics.csv')
        logger.info(f'Validation metrics are being written to:  {os.path.join(path, prefix)}_validation_metrics.csv')
        logger.info(f'Weights are being written to: {os.path.join(path, "weights")}')
        for i in range(num_epochs):
                logger.info(f"Epoch {i}")
                projector.train()
                model.train()
                optimizer.zero_grad()
                processed_p_emb, processed_g_emb = projector(amp_embeddings, target_embeddings)
                pos_scores, neg_scores = model(processed_p_emb, processed_g_emb, graphs[0], graphs[1], graphs[2], device)

                loss = loss_bce_weighted(pos_scores, neg_scores, device)
                calc_metrics(graphs[1].edges(etype='is_active')[1], graphs[2].edges(etype='is_active')[1], 
                                                        pos_scores, neg_scores, tacc, tf1, tmf1, tprecision, trecall, tmcc, tauroc, tavp)
                losses.append(loss.item())
                
                loss.backward()
                optimizer.step()
                if scheduler_step_size != "":
                        scheduler.step()

                model.eval()
                projector.eval()
                with torch.no_grad():
                        processed_p_emb, processed_g_emb = projector(amp_embeddings, target_embeddings)
                        vpos_scores, vneg_scores = model(processed_p_emb, processed_g_emb, graphs[3], graphs[4], graphs[5], device)
                        calc_metrics(graphs[4].edges(etype='is_active')[1], graphs[5].edges(etype='is_active')[1], 
                                                        vpos_scores, vneg_scores, vacc, vf1, vmf1, vprecision, vrecall, vmcc, vauroc, vavp)
                        loss = loss_bce_unweighted(vpos_scores, vneg_scores, device)
                        vlosses.append(loss.item())

                if early_stopper.check(vlosses[i]):
                        break   
                
                plot_metrics(path, prefix, learning_rate, scheduler_step_size, scheduler_gamma,
                        losses, vlosses, tacc, vacc, 
                        tf1, vf1, tmf1, vmf1, tprecision, vprecision, 
                        trecall, vrecall)
                
                training_df = pd.DataFrame({"Epoch":np.arange(len(tacc)), "Accuracy":tacc, "Precision":tprecision, "Recall": trecall, "F1": tf1, 'MacroF1':tmf1, "MCC": tmcc, "AUROC": tauroc, "AVP": tavp})
                training_df.to_csv(os.path.join(path, f"{prefix}_train_metrics.csv"))
                validation_df = pd.DataFrame({"Epoch":np.arange(len(vacc)), "Accuracy":vacc, "Precision":vprecision, "Recall": vrecall, "F1": vf1, 'MacroF1':vmf1, "MCC": vmcc, "AUROC": vauroc, "AVP": vavp})
                validation_df.to_csv(os.path.join(path, f"{prefix}_validation_metrics.csv"))

                torch.save(model.state_dict(), os.path.join(path, "weights", f"{prefix}_triAMPh_weight_{i}.pth"))
                torch.save(projector.state_dict(), os.path.join(path, "weights", f"{prefix}_projector_weight_{i}.pth"))
                
                torch.cuda.empty_cache()
        
        return True


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
                if best_dict["Learning Scheme"] == "inductive":
                        processed_p_emb, processed_g_emb = projector(data.amp_embeddings[graph_idx//3].to(device), data.target_embeddings.to(device))
                else:
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
        
        # Data splitting arguments
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
                default = 50
        )
        parser.add_argument(
                "--inductive",
                help="Training strategy: Inductive if 1, transductive otherwise.",
                type=int,
                required=False,
                default = 1
        )
        
        # Training optimization arguments
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
                default = 2000
        )

        parser.add_argument(
                "--dropout",
                help="Dropout percentage for Heterogeneous Graph Attention Network.",
                type=int,
                required=False,
                default = 50
        )

        parser.add_argument(
                "--seed",
                help="Random seed to be set.",
                type=int,
                required=False,
                default = 123
        )
        
        parser.add_argument(
                "--patience",
                help="Early stopper patience.",
                type=int,
                required=False,
                default = 200
        )

        parser.add_argument(
                "--epsilon",
                help="Early stopper epsilon.",
                type=float,
                required=False,
                default = 0.01
        )

        # Architecture arguments
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
                "--semantic_att_size",
                help="The size of the semantic attention vector.",
                type=int,
                required=False,
                default = 128
        )
        
        parser.add_argument(
                "--self_emb_semantic",
                help="The strategy that will be used for self embedding inclusion. 1 for semantic attention level, otherwise graph attention level.",
                type=int,
                required=False,
                default = 0
        )
        
        parser.add_argument(
                "--p_percentile",
                help="The percentile cutoff value for pairwise peptide distances to be used for graph building.",
                type=int,
                required=False,
                default = 15
        )
        
        parser.add_argument(
                "--g_percentile",
                help="The percentile cutoff value for pairwise genome distances to be used for graph building.",
                type=int,
                required=False,
                default = 30
        )

        return parser.parse_args()


def main():
        args = get_args()

        test_split = 100 - (args.tr_split + args.val_split)
        if test_split != 0 :
                splits = [args.tr_split/100, args.val_split/100, test_split/100]
        else:
                splits = [args.tr_split/100, args.val_split/100]
        test = test_split != 0

        if args.inductive:
                data = triAMPhInductiveData(args.positive_edges,
                                args.negative_edges,
                                args.protein_emb_dir,
                                args.genomic_emb_dir,
                                splits, 
                                args.msg_pas/100)
                learning_scheme = "inductive"
                
        else:
                data = triAMPhTransductiveData(args.positive_edges,
                                args.negative_edges,
                                args.protein_emb_dir,
                                args.genomic_emb_dir,
                                splits, 
                                args.msg_pas/100)
                learning_scheme = "transductive"
        
        logger.info("Graphs have been constructed according to the following parameters.")
        logger.info(f"Learning strategy: {learning_scheme}")
        logger.info(f"Positive edge file: {args.positive_edges}")
        logger.info(f"Negative edge file: {args.negative_edges}")
        if test:
                logger.info(f"Train, validation, test split (%): {splits}")
        else:
                logger.info(f"Train & validation split (%): {splits}")
        
        logger.info(f"Message passing proportion (%): {args.msg_pas}")
                

        stop = args.patience
        if stop > args.epochs:
                logger.warning(f"Early stopper patience ({args.patience}) is bigger than the number of epochs ({args.epochs}). Early stopping patience is being set to {int(args.epochs/3)} instead.")
                stop = int(args.epochs/3)

        
        # ATTENTION: the order of the metapaths during the test stage should be the same
        meta_paths = [["is_active", "is_similar_g"],["is_susceptable", "is_similar_p"], ["is_similar_p", "is_active"], ["is_similar_g", "is_susceptable"], ["is_active", "is_susceptable"], ["is_susceptable", "is_active"]]

        logger.info("Training is starting with the following parameters")
        logger.info(f"Meta paths: {meta_paths}")
        logger.info(f"Peptide embedding distance cutoff (percentile): {args.p_percentile}")
        logger.info(f"Genome embedding distance cutoff (percentile): {args.g_percentile}")
        if args.self_emb_semantic==1:
                logger.info("Self embeddings are being incorporated in semantic attention stage.")
        else:
                logger.info("Self embeddings are being incorporated in graph attention stage.")

        logger.info(f"Output directory is set to: {args.output_dir}")
        
        if args.inductive:
                train_inductive(meta_paths, args.han_input_size, args.han_hidden_size, args.n_heads, args.dropout/100, args.semantic_att_size, args.p_percentile, args.g_percentile,
                                args.lr, "", "", args.epsilon, stop, args.epochs, 
                                args.output_dir, args.prefix,  
                                args.msg_pas, learning_scheme, data.graphs, data.amp_embeddings, data.target_embeddings,
                                args.self_emb_semantic==1, args.self_emb_semantic!=1,
                                args.prot_emb_size, args.gen_emb_size, args.seed)
        else:
                train_transductive(meta_paths, args.han_input_size, args.han_hidden_size, args.n_heads, args.dropout/100, args.semantic_att_size, args.p_percentile, args.g_percentile,
                                args.lr, "", "", args.epsilon, stop, args.epochs, 
                                args.output_dir, args.prefix,  
                                args.msg_pas, learning_scheme, data.graphs, data.amp_embeddings, data.target_embeddings,
                                args.self_emb_semantic==1, args.self_emb_semantic!=1,
                                args.prot_emb_size, args.gen_emb_size, args.seed)
        
        if test:
                predict_based_on_best_validation(args.output_dir, args.prefix, data, 6)
        
if __name__ == "__main__":
        main()