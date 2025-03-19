import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from dgl import heterograph
from torch import randint, from_numpy
from torch.nn.utils.rnn import pad_sequence
import torch
import utils
import constants
import os
import gc

class triAMPhDataTest:
    def __init__(self, 
                positive_amp_target_path:str,
                negative_amp_target_path: str, 
                positive_test_path: str,
                negative_test_path: str,
                amp_embedding_path:str,
                target_embedding_path:str,
                triple_stats: bool = False
                ):
        
        self.positive_amp_target_path = positive_amp_target_path
        self.negative_amp_target_path = negative_amp_target_path
        self.positive_test_path = positive_test_path
        self.negative_test_path = negative_test_path
        self.amp_embedding_path = amp_embedding_path
        self.target_embedding_path = target_embedding_path
        
        (self.message_passing_adj, self.test_adj, self.neg_test_adj, self.amp_idx, self.amp_embeddings, self.amp_masks,
            self.target_idx, self.target_embeddings, self.target_masks) = self.get_adjacency_matrices(triple_stats)
        self.graphs = self.build_graph_and_split_data()

    def get_adjacency_matrices(self, triple_stats): # this assumes if there is a pathogen involved, there are both negative and positive edges for it!
        amp_target = pd.read_csv(self.positive_amp_target_path)
        neg_amp_target = pd.read_csv(self.negative_amp_target_path)
        amp_target = amp_target[["ID", "Pathogens", "Sequence"]]
        neg_amp_target = neg_amp_target[["ID", "Pathogens", "Sequence"]]
        amp_target["Merged"] = amp_target["Sequence"] + amp_target["Pathogens"]
        neg_amp_target["Merged"] = neg_amp_target["Sequence"] + neg_amp_target["Pathogens"]
        
        amp_embeddings=os.listdir(self.amp_embedding_path)
        amp_embeddings=[x.split('.n')[0] for x in amp_embeddings]
        
        target_embeddings=os.listdir(self.target_embedding_path)
        target_embeddings=[x.split('.')[0] for x in target_embeddings]

        test = pd.read_csv(self.positive_test_path)
        neg_test = pd.read_csv(self.negative_test_path)
        test = test[["ID", "Pathogens", "Sequence"]]
        neg_test = neg_test[["ID", "Pathogens", "Sequence"]]
        test["Merged"] = test["Sequence"] + test["Pathogens"]
        neg_test["Merged"] = neg_test["Sequence"] + neg_test["Pathogens"]

        # ensure the test sequences are independent, the ones also occuring in the training set is removed
        test = test.loc[~(test["Merged"].isin(amp_target["Merged"].to_numpy())) & ~(test["Merged"].isin(neg_amp_target["Merged"].to_numpy()))]
        neg_test = neg_test.loc[~(neg_test["Merged"].isin(neg_amp_target["Merged"].to_numpy()))&~(neg_test["Merged"].isin(amp_target["Merged"].to_numpy()))]
        if len(test) == 0:
            print("No positive independent test peptide-pathogen pairs!")
            return None
        elif len(neg_test) == 0:
            print("No negative independent test peptide-pathogen pairs!")
            return None 


        # since there might be some matchings in amp-target file that does not have an embedding for either an amp or a target
        # we need to eliminate the lines that suffers from either on of the conditions
        amp_target["ID"] = amp_target["ID"].astype("str")
        amp_target = amp_target.loc[(amp_target["ID"].isin(amp_embeddings))]
        neg_amp_target["ID"] = neg_amp_target["ID"].astype("str")
        neg_amp_target = neg_amp_target.loc[(neg_amp_target["ID"].isin(amp_embeddings))]
        amp_target = amp_target.loc[amp_target["Pathogens"].isin(target_embeddings)] # amp_target = amp_target.loc[amp_target["Pathogens"].isin(target_embeddings.keys())]
        neg_amp_target = neg_amp_target.loc[neg_amp_target["Pathogens"].isin(target_embeddings)] # amp_target = amp_target.loc[amp_target["Pathogens"].isin(target_embeddings.keys())]

        test["ID"] = test["ID"].astype("str")
        test = test.loc[(test["ID"].isin(amp_embeddings))]
        neg_test["ID"] = neg_test["ID"].astype("str")
        neg_test = neg_test.loc[(neg_test["ID"].isin(amp_embeddings))]
        
        test = test.loc[test["Pathogens"].isin(target_embeddings)] # amp_target = amp_target.loc[amp_target["Pathogens"].isin(target_embeddings.keys())]
        neg_test = neg_test.loc[neg_test["Pathogens"].isin(target_embeddings)] # 
       
        # check vice versa as well
        amp_embeddings = {}
        tmp = np.unique(np.concat([amp_target["ID"].to_numpy(), neg_amp_target["ID"].to_numpy(), test["ID"].to_numpy(), neg_test["ID"].to_numpy()]))
        print(f"Number of AMPs in the feature matrix:{len(tmp)}")
        for amp in tmp:
            amp_embeddings[amp] = np.load(os.path.join(self.amp_embedding_path, f"{amp}.npy"))
        
        target_embeddings = {}
        tmp = np.unique(np.concat([amp_target["Pathogens"].to_numpy(), neg_amp_target["Pathogens"].to_numpy(), test["Pathogens"].to_numpy(), neg_test["Pathogens"].to_numpy()]))
        print(f"Number of Targets in the feature matrix:{len(tmp)}")
        for target in tmp:
            target_embeddings[target] = np.load(os.path.join(self.target_embedding_path, f"{target}.npy"))
        
        print(f"Number of AMP-Target Pairs for the Message Passing Graph: {len(amp_target)}")
        print(f"Number of AMP-Target Pairs for the Positive Test Graph: {len(test)}")
        print(f"Number of AMP-Target Pairs for the Negative Test Graph: {len(neg_test)}")
        print(f"Number of AMPs in positive training: {len(amp_target['ID'])}")
        print(f"Number of AMPs in positive test: {len(test['ID'])}")
        print(f"Number of AMPs in negative test: {len(neg_test['ID'])}")
        print(f"Number of AMPs in total (unique): {len(np.unique(np.concatenate([amp_target['ID'].to_numpy(), neg_amp_target['ID'].to_numpy(), test['ID'].to_numpy(), neg_test['ID'].to_numpy()])))}")

        # -------- Message Passing Adjacency Matrix --------
        # reaggregate to group based on AMP IDs
        # since we might have peptides that are not active against any pathogen, we need to add those as empty pathogen columns as well
        empty_rows_mp = neg_amp_target[~neg_amp_target["ID"].isin(amp_target["ID"].to_numpy())].drop_duplicates( subset ='ID', keep = 'last').reset_index(drop = True) 
        empty_rows_mp["Pathogens"] = "" # empty as they are negative edges!
        # test peptide-pathogen pairs have to be annotated as empyt as well
        # we have to have all the peptides mentioned in the test in the graph as well 
        test_pep = pd.concat([test, neg_test])
        test_pep["Pathogens"] = ""
        test_pep = test_pep.drop_duplicates(subset="ID")
        message_passing_adj = pd.concat([amp_target, empty_rows_mp, test_pep]).reset_index(drop = True)[["ID", "Pathogens"]]
        message_passing_adj = message_passing_adj.groupby(["ID"]).agg({"Pathogens": lambda x: x.tolist()})
        message_passing_adj = pd.get_dummies(message_passing_adj["Pathogens"].explode()).groupby(level=0).sum().drop(columns="")
        # add missing columns
        for target in target_embeddings.keys():
            if target not in message_passing_adj.columns:
                message_passing_adj[target]=0

        
        # -------- Supervision Adjacency Matrix --------
        # reaggregate to group based on AMP IDs
        # since we might have peptides that are not active against any pathogen, we need to add those as empty pathogen columns as well
        empty_rows_test = neg_test[~neg_test["ID"].isin(test["ID"].to_numpy())].drop_duplicates( subset = "ID", keep = 'last').reset_index(drop = True) 
        empty_rows_test["Pathogens"] = "" # empty as they are negative edges!
        # test peptide-pathogen pairs have to be annotated as empyt as well
        # we have to have all the peptides mentioned in the test in the graph as well 
        amp_target["Pathogens"] = ""
        empty_rows_mp["Pathogens"] = ""
        mp = pd.concat([empty_rows_mp, amp_target])
        mp = mp.drop_duplicates( subset = "ID", keep = 'last')
        test_adj = pd.concat([test, empty_rows_test, mp]).reset_index(drop = True)[["ID", "Pathogens"]]
        test_adj = test_adj.groupby(["ID"]).agg({"Pathogens": lambda x: x.tolist()})
        test_adj = pd.get_dummies(test_adj["Pathogens"].explode()).groupby(level=0).sum().drop(columns="")
        # add missing columns
        for target in target_embeddings.keys():
            if target not in test_adj.columns:
                test_adj[target]=0

        #same for negative test
        empty_rows_neg = test[~test["ID"].isin(neg_test["ID"].to_numpy())].drop_duplicates( subset = "ID", keep = 'last').reset_index(drop = True) 
        empty_rows_neg["Pathogens"] = "" # empty as they are positive edges!
        # test peptide-pathogen pairs have to be annotated as empyt as well
        # we have to have all the peptides mentioned in the test in the graph as well 
        neg_test_adj = pd.concat([neg_test, empty_rows_neg, mp]).reset_index(drop = True)[["ID", "Pathogens"]]
        neg_test_adj = neg_test_adj.groupby(["ID"]).agg({"Pathogens": lambda x: x.tolist()})
        neg_test_adj = pd.get_dummies(neg_test_adj["Pathogens"].explode()).groupby(level=0).sum().drop(columns="")
        for target in target_embeddings.keys():
            if target not in neg_test_adj.columns:
                neg_test_adj[target]=0
        
        # Arrange columns and rows so that we have consistant indices and IDs 
        message_passing_adj = message_passing_adj.reindex(columns=target_embeddings.keys(), index=amp_embeddings.keys())
        message_passing_adj = message_passing_adj.to_numpy()
        test_adj = test_adj.reindex(columns=target_embeddings.keys(), index=amp_embeddings.keys())
        test_adj = test_adj.to_numpy()
        neg_test_adj = neg_test_adj.reindex(columns=target_embeddings.keys(), index=amp_embeddings.keys())
        neg_test_adj = neg_test_adj.to_numpy()
        

        # indices - consistent for amp & target pairs too
        amp_idx = {value: index for index, value in enumerate(amp_embeddings.keys())}
        target_idx = {value: index for index, value in enumerate(target_embeddings.keys())}
        
        print(f"Dimensions of Message Passing AMP-Target Adjacency Matrix: {message_passing_adj.shape}")
       
        
        if not triple_stats:
            amp_embeddings = [torch.mean(from_numpy(value), dim=0) for value in amp_embeddings.values()]
            target_embeddings = [torch.mean(from_numpy(value).view(-1, 512), dim=0) for value in target_embeddings.values()] # TODO automatic size instead of 512
        
        else:
            qs = torch.tensor([0.25, 0.5, 0.75])
            amp_embeddings = [torch.quantile(from_numpy(value), qs, dim=0).flatten() for value in amp_embeddings.values()]
            target_embeddings = [torch.quantile(from_numpy(value).view(-1, 512), qs, dim=0).flatten() for value in target_embeddings.values()]

        amp_embeddings = torch.stack(amp_embeddings, dim=0)
        amp_masks = torch.empty((amp_embeddings.size()[0], amp_embeddings.size()[0]))

        target_embeddings = torch.stack(target_embeddings, dim=0)
        target_masks = torch.empty((target_embeddings.size()[0], target_embeddings.size()[0]))
        
        del empty_rows_mp
        del empty_rows_test
        del empty_rows_neg
        del neg_amp_target
        del amp_target
        del test
        del neg_test
        gc.collect()
        return (csr_matrix(message_passing_adj), csr_matrix(test_adj), csr_matrix(neg_test_adj), amp_idx, amp_embeddings, amp_masks, target_idx, target_embeddings, target_masks) 

    def build_graph_and_split_data(self):
        """ generates the message passing, positive test and negative test graphs for the test data """
        num_nodes_dict = {'AMP': self.amp_embeddings.size()[0], 'Target': self.target_embeddings.size()[0]}
        edges = self.message_passing_adj.nonzero()
        messaga_passing_graph = heterograph({
            ("AMP", "is_similar_p", "AMP"): ([0], [0]),
            ("Target", "is_similar_g", "Target"): ([0], [0]),
            ("AMP", "is_active", "Target"): (edges[0], edges[1]),
            ("Target", "is_susceptable", "AMP"): (edges[1], edges[0]) 
        }, num_nodes_dict=num_nodes_dict)
        mes_pass_edges=[]
        for i in range(len(edges[0])):
            mes_pass_edges.append(f'{edges[0][i]}, {edges[1][i]}')
        
        edges = self.test_adj.nonzero()
        positive_graph = heterograph({
            ("AMP", "is_active", "Target"): (edges[0], edges[1]),
            ("Target", "is_susceptable", "AMP"): (edges[1], edges[0]) 
        }, num_nodes_dict=num_nodes_dict)
        pos_test_edges=[]
        for i in range(len(edges[0])):
            pos_test_edges.append(f'{edges[0][i]}, {edges[1][i]}')
        
        edges = self.neg_test_adj.nonzero()
        negative_graph = heterograph({
            ("AMP", "is_active", "Target"): (edges[0], edges[1]),
            ("Target", "is_susceptable", "AMP"): (edges[1], edges[0]) 
        }, num_nodes_dict=num_nodes_dict)
        neg_test_edges=[]
        for i in range(len(edges[0])):
            neg_test_edges.append(f'{edges[0][i]}, {edges[1][i]}')
        graphs = [messaga_passing_graph, positive_graph, negative_graph]
        
        mes_pass_edges = set(mes_pass_edges)
        pos_test_edges=set(pos_test_edges)
        neg_test_edges=set(neg_test_edges)

        return graphs

# ===========================================================================================================================
class triAMPDataInductive:
    def __init__(self, 
                positive_amp_target_path:str,
                negative_amp_target_path: str, 
                amp_embedding_path:str,
                target_embedding_path:str,
                data_split: list,
                mes_passing = 0.8,
                triple_stats = False,
                balance_negative: bool = False
                ):
        
        self.positive_amp_target_path = positive_amp_target_path
        self.negative_amp_target_path = negative_amp_target_path
        self.amp_embedding_path = amp_embedding_path
        self.target_embedding_path = target_embedding_path
        self.data_split = data_split # [train_proportion, validation_proportion, test_proportion]
        
        (self.amp_target_adj, self.neg_amp_target_adj, self.amp_idx, self.amp_embeddings, self.amp_masks,
            self.target_idx, self.target_embeddings, self.target_masks) = self.get_adjacency_matrices(triple_stats)
        self.graphs, self.amp_embeddings = self.build_graph_and_split_data(mes_passing, balance_negative)
        gc.collect()
        

    def get_adjacency_matrices(self,triple_stats): # this assumes if there is a pathogen involved, there are both negative and positive edges for it!
        amp_target = pd.read_csv(self.positive_amp_target_path)
        neg_amp_target = pd.read_csv(self.negative_amp_target_path)

        amp_embeddings=os.listdir(self.amp_embedding_path)
        amp_embeddings=[x.split('.n')[0] for x in amp_embeddings]
        
        target_embeddings=os.listdir(self.target_embedding_path)
        target_embeddings=[x.split('.')[0] for x in target_embeddings]

        # since there might be some matchings in amp-target file that does not have an embedding for either an amp or a target
        # we need to eliminate the lines that suffers from either on of the conditions
        print(f"Number of AMPs in folder:{len(set(amp_embeddings))}")
        print(f"Number of AMPs in the positive link file:{len(set(amp_target['ID'].to_numpy()))}")
        print(f"Number of AMPs in the negative link file:{len(set(neg_amp_target['ID'].to_numpy()))}")
        print(f"Number of AMPs in total (unique): {len(np.unique(np.append(amp_target['ID'].to_numpy(), neg_amp_target['ID'].to_numpy())))}")

        amp_target = amp_target.loc[(amp_target["ID"].isin(amp_embeddings))]
        neg_amp_target = neg_amp_target.loc[(neg_amp_target["ID"].isin(amp_embeddings))]
        amp_target = amp_target.loc[amp_target["Pathogens"].isin(target_embeddings)] # amp_target = amp_target.loc[amp_target["Pathogens"].isin(target_embeddings.keys())]
        neg_amp_target = neg_amp_target.loc[neg_amp_target["Pathogens"].isin(target_embeddings)] # amp_target = amp_target.loc[amp_target["Pathogens"].isin(target_embeddings.keys())]

        # check vice versa as well
        amp_embeddings = {}
        tmp = np.unique(np.append(amp_target["ID"].to_numpy(), neg_amp_target["ID"].to_numpy()))
        print(f"Number of AMPs in the feature matrix:{len(tmp)}")
        for amp in tmp:
            amp_embeddings[amp] = np.load(os.path.join(self.amp_embedding_path, f"{amp}.npy"))
        
        target_embeddings = {}
        tmp = np.unique(np.append(amp_target["Pathogens"].to_numpy(), neg_amp_target["Pathogens"].to_numpy()))
        print(f"Number of Targets in the feature matrix:{len(tmp)}")
        for target in tmp:
            target_embeddings[target] = np.load(os.path.join(self.target_embedding_path, f"{target}.npy"))
        
        # reaggregate to group based on AMP IDs
        # since we might have peptides that are not active against any pathogen, we need to add those as empty pathogen columns as well
        empty_rows = neg_amp_target[~neg_amp_target["ID"].isin(amp_target["ID"].to_numpy())].drop_duplicates( subset = ['Sequence',"ID"], keep = 'last').reset_index(drop = True) 
        empty_rows.loc[:,"Pathogens"] = "" # empty as they are negative edges!
        print(f"Empty rows len pos: {len(empty_rows)}")
        amp_target = pd.concat([amp_target, empty_rows]).reset_index(drop = True) 
        amp_target_tmp = pd.concat([amp_target, empty_rows]) # TODO PLEASE CHECK THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        amp_target = amp_target.groupby(["ID"]).agg({"Pathogens": lambda x: x.tolist()})
        amp_target_adj = pd.get_dummies(amp_target["Pathogens"].explode()).groupby(level=0).sum()
        
        # same thing applies for the negative edges
        empty_rows = amp_target_tmp[~amp_target_tmp["ID"].isin(neg_amp_target["ID"].to_numpy())].drop_duplicates( subset = ['Sequence',"ID"], keep = 'last').reset_index(drop = True) 
        empty_rows.loc[:,"Pathogens"] = "" # empty as they are negative edges!
        print(f"Empty rows len neg: {len(empty_rows)}")
        neg_amp_target = pd.concat([neg_amp_target,empty_rows])
        neg_amp_target = neg_amp_target.groupby(["ID"]).agg({"Pathogens": lambda x: x.tolist()})
        neg_amp_target_adj = pd.get_dummies(neg_amp_target["Pathogens"].explode()).groupby(level=0).sum()
        
        # AMP to Target or Target to AMP:
        amp_target_adj = amp_target_adj.reindex(columns=target_embeddings.keys(), index=amp_embeddings.keys())
        amp_target_adj = amp_target_adj.to_numpy()
        neg_amp_target_adj = neg_amp_target_adj.reindex(columns=target_embeddings.keys(), index=amp_embeddings.keys())
        neg_amp_target_adj = neg_amp_target_adj.to_numpy()

        # indices - consistent for amp & target pairs too
        amp_idx = {value: index for index, value in enumerate(amp_embeddings.keys())}
        self.amp_converter = {index: value for index, value in enumerate(amp_embeddings.keys())}
        target_idx = {value: index for index, value in enumerate(target_embeddings.keys())}
        
        print(f"Dimensions of AMP-Target Adjacency Matrix: {amp_target_adj.shape}")
        print(f"Dimensions of Negative AMP-Target Adjacency Matrix: {neg_amp_target_adj.shape}")
        print(f"Number of AMPs in the feature matrix:{len(amp_embeddings)}")
        print(f"Number of Targets in the feature matrix:{len(target_embeddings)}")
       
        if not triple_stats:
            amp_embeddings = [torch.mean(from_numpy(value), dim=0) for value in amp_embeddings.values()]
            target_embeddings = [torch.mean(from_numpy(value).view(-1, 512), dim=0) for value in target_embeddings.values()] # TODO automatic size instead of 512
        
        else:
            qs = torch.tensor([0.25, 0.5, 0.75])
            amp_embeddings = [torch.quantile(from_numpy(value), qs, dim=0).flatten() for value in amp_embeddings.values()]
            target_embeddings = [torch.quantile(from_numpy(value).view(-1, 512), qs, dim=0).flatten() for value in target_embeddings.values()]

        amp_embeddings = torch.stack(amp_embeddings, dim=0)
        amp_masks = torch.empty((amp_embeddings.size()[0], amp_embeddings.size()[0]))

        target_embeddings = torch.stack(target_embeddings, dim=0)
        target_masks = torch.empty((target_embeddings.size()[0], target_embeddings.size()[0]))
        
        return (csr_matrix(amp_target_adj), csr_matrix(neg_amp_target_adj), amp_idx, amp_embeddings, amp_masks, target_idx, target_embeddings, target_masks) 

    def build_graph_and_split_data(self, mes_passing, balance_negative):
        # here instead of splitting edges, we split nodes --> inductive learning, but only peptide nodes
        utils.set_seed(constants.SEED)
        node_cnt = self.amp_target_adj.shape[0]
        val_cnt = int(node_cnt * self.data_split[1])
        train_cnt = node_cnt - val_cnt
        print(f'Node count: {node_cnt}')
        if len(self.data_split) == 3:
            test_cnt = int(node_cnt * self.data_split[2])
            train_cnt = train_cnt - test_cnt
        print(f'Training count: {train_cnt}')
        print(f'Validation count: {val_cnt}')
        
        # split the nodes based on the counts calculated above
        nids = np.arange(node_cnt)
        nids = np.random.permutation(nids)

        train_nodes = nids[:train_cnt]
        self.train_nodes = train_nodes
        val_nodes = nids[train_cnt:train_cnt+val_cnt]
        self.val_nodes = val_nodes
        if len(self.data_split) == 3:
            test_nodes = nids[train_cnt+val_cnt:]
            self.test_nodes = test_nodes

        edges = self.amp_target_adj.nonzero()
        neg_edges = self.neg_amp_target_adj.nonzero()

        # split edges based on the nodes we defined for each split
        mask = np.isin(edges[0], train_nodes)
        training_edges = (edges[0][mask], edges[1][mask])
        mask = np.isin(edges[0], val_nodes)
        val_edges = (edges[0][mask], edges[1][mask])
        if len(self.data_split) == 3:
            mask = np.isin(edges[0], test_nodes)
            test_edges = (edges[0][mask], edges[1][mask])

        # same applies for the negative graphs
        mask = np.isin(neg_edges[0], train_nodes)
        neg_training_edges = (neg_edges[0][mask], neg_edges[1][mask])
        mask = np.isin(neg_edges[0], val_nodes)
        neg_val_edges = (neg_edges[0][mask], neg_edges[1][mask])
        if len(self.data_split) == 3:
            mask = np.isin(neg_edges[0], test_nodes)
            neg_test_edges = (neg_edges[0][mask], neg_edges[1][mask])

        # now we need to disjoin the message passing and supervision edges (BE CAREFUL, EDGES ARE BEING SEPARATED HERE)
        eids = np.arange(len(training_edges[0]))
        eids = np.random.permutation(eids) 
        mp_cnt = int(len(training_edges[0]) * mes_passing)
        train_mp_u, train_mp_v = training_edges[0][eids[:mp_cnt]], training_edges[1][eids[:mp_cnt]]
        train_u, train_v =  training_edges[0][eids[mp_cnt:]], training_edges[1][eids[mp_cnt:]]

        eids = np.arange(len(val_edges[0]))
        eids = np.random.permutation(eids) 
        mp_cnt = int(len(val_edges[0]) * mes_passing)
        val_mp_u, val_mp_v = val_edges[0][eids[:mp_cnt]], val_edges[1][eids[:mp_cnt]]
        val_u, val_v =  val_edges[0][eids[mp_cnt:]], val_edges[1][eids[mp_cnt:]]
        
        if len(self.data_split) == 3:
            eids = np.arange(len(test_edges[0]))
            eids = np.random.permutation(eids) 
            mp_cnt = int(len(test_edges[0]) * mes_passing)
            test_mp_u, test_mp_v = test_edges[0][eids[:mp_cnt]], test_edges[1][eids[:mp_cnt]]
            test_u, test_v =  test_edges[0][eids[mp_cnt:]], test_edges[1][eids[mp_cnt:]]

        # control leaks
        train_mp_set = set([str(u)+','+str(train_mp_v[i]) for i, u in enumerate(train_mp_u)])
        train_pos_set = set([str(u)+','+str(train_v[i]) for i, u in enumerate(train_u)])
        train_neg_set = set([str(u)+','+str(neg_training_edges[1][i]) for i, u in enumerate(neg_training_edges[0])])

        val_mp_set = set([str(u)+','+str(val_mp_v[i]) for i, u in enumerate(val_mp_u)])
        val_pos_set = set([str(u)+','+str(val_v[i]) for i, u in enumerate(val_u)])
        val_neg_set = set([str(u)+','+str(neg_val_edges[1][i]) for i, u in enumerate(neg_val_edges[0])])

        test_mp_set = set([str(u)+','+str(test_mp_v[i]) for i, u in enumerate(test_mp_u)])
        test_pos_set = set([str(u)+','+str(test_v[i]) for i, u in enumerate(test_u)])
        test_neg_set = set([str(u)+','+str(neg_test_edges[1][i]) for i, u in enumerate(neg_test_edges[0])])


        # here the keys are the og ids and the values are the new ids!!! - it's only for the peptides as we include all the pathogens
        train_dict = {}
        self.train_pep = {}
        i = 0
        for node in train_nodes:
            train_dict[node] = i 
            self.train_pep[self.amp_converter[node]] = i 
            i+=1

        val_dict = {}
        self.val_pep = {}
        i = 0
        for node in val_nodes:
            val_dict[node] = i 
            self.val_pep[self.amp_converter[node]] = i 
            i+=1
        
        
        # convert global peptide ids to local ids
        train_mp_u = [ train_dict[node] for node in train_mp_u]
        train_u = [ train_dict[node] for node in train_u]
        neg_train_u = [ train_dict[node] for node in neg_training_edges[0]]
        neg_train_v = neg_training_edges[1]
        if balance_negative:
            neg_train_u = neg_train_u[:len(train_u)]
            neg_train_v = neg_train_v[:len(train_u)]
        
        val_mp_u = [ val_dict[node] for node in val_mp_u]
        val_u = [ val_dict[node] for node in val_u]
        neg_val_u = [ val_dict[node] for node in neg_val_edges[0]]


        num_nodes_dict = {'AMP': len(set(train_nodes)), 'Target': len(set(edges[1]))}
        print(num_nodes_dict)
        train_graph = heterograph({
            ("AMP", "is_similar_p", "AMP"): ([0], [0]),
            ("Target", "is_similar_g", "Target"): ([0], [0]),
            ("AMP", "is_active", "Target"): (train_mp_u, train_mp_v),
            ("Target", "is_susceptable", "AMP"): (train_mp_v, train_mp_u) 
        }, num_nodes_dict=num_nodes_dict)
        train_graph_supervision = heterograph({
            ("AMP", "is_active", "Target"): (train_u, train_v)
        }, num_nodes_dict=num_nodes_dict)
        train_graph_neg = heterograph({
            ("AMP", "is_active", "Target"): (neg_train_u, neg_train_v)
        }, num_nodes_dict=num_nodes_dict)

        num_nodes_dict = {'AMP': len(set(val_nodes)), 'Target': len(set(edges[1]))}
        val_graph = heterograph({
            ("AMP", "is_similar_p", "AMP"): ([0], [0]),
            ("Target", "is_similar_g", "Target"): ([0], [0]),
            ("AMP", "is_active", "Target"): (val_mp_u, val_mp_v),
            ("Target", "is_susceptable", "AMP"): (val_mp_v, val_mp_u) 
        }, num_nodes_dict=num_nodes_dict)
        val_graph_supervision = heterograph({
            ("AMP", "is_active", "Target"): (val_u, val_v)
        }, num_nodes_dict=num_nodes_dict)
        val_graph_neg = heterograph({
            ("AMP", "is_active", "Target"): (neg_val_u, neg_val_edges[1])
        }, num_nodes_dict=num_nodes_dict)
     

        if len(self.data_split) == 3:
            test_dict = {}
            self.test_pep = {}
            i = 0
            for node in test_nodes:
                test_dict[node] = i 
                self.test_pep[self.amp_converter[node]] = i 
                i+=1

            # convert global peptide ids to local ids
            test_mp_u = [ test_dict[node] for node in test_mp_u]
            test_u = [ test_dict[node] for node in test_u]
            neg_test_u = [ test_dict[node] for node in neg_test_edges[0]]

            num_nodes_dict = {'AMP': len(set(test_nodes)), 'Target': len(set(edges[1]))}
            test_graph = heterograph({
                ("AMP", "is_similar_p", "AMP"): ([0], [0]),
                ("Target", "is_similar_g", "Target"): ([0], [0]),
                ("AMP", "is_active", "Target"): (test_mp_u, test_mp_v),
                ("Target", "is_susceptable", "AMP"): (test_mp_v, test_mp_u) 
            }, num_nodes_dict=num_nodes_dict)
            test_graph_supervision = heterograph({
                ("AMP", "is_active", "Target"): (test_u, test_v)
            }, num_nodes_dict=num_nodes_dict)
            test_graph_neg = heterograph({
                ("AMP", "is_active", "Target"): (neg_test_u, neg_test_edges[1])
            }, num_nodes_dict=num_nodes_dict)
            graphs = [train_graph, train_graph_supervision, train_graph_neg, val_graph, val_graph_supervision, val_graph_neg, test_graph, test_graph_supervision, test_graph_neg]
            amp_embs = [self.amp_embeddings[np.sort(train_nodes),:], self.amp_embeddings[np.sort(val_nodes),:], self.amp_embeddings[np.sort(test_nodes),:]]
        else:
            graphs = [train_graph, train_graph_supervision, train_graph_neg, val_graph, val_graph_supervision, val_graph_neg]
            amp_embs = [self.amp_embeddings[np.sort(train_nodes),:], self.amp_embeddings[np.sort(val_nodes),:]]
        return graphs, amp_embs

# ===========================================================================================================================
class triAMPData:
    def __init__(self, 
                positive_amp_target_path:str,
                negative_amp_target_path: str, 
                amp_embedding_path:str,
                target_embedding_path:str,
                data_split: list,
                mes_passing = 0.8,
                triple_stats = False,
                balance_negative = False
                ):
        
        self.positive_amp_target_path = positive_amp_target_path
        self.negative_amp_target_path = negative_amp_target_path
        self.amp_embedding_path = amp_embedding_path
        self.target_embedding_path = target_embedding_path
        self.data_split = data_split # [train_proportion, validation_proportion, test_proportion]
        
        (self.amp_target_adj, self.neg_amp_target_adj, self.amp_idx, self.amp_embeddings, self.amp_masks,
            self.target_idx, self.target_embeddings, self.target_masks) = self.get_adjacency_matrices(triple_stats)
        self.graphs = self.build_graph_and_split_data(mes_passing, balance_negative)
        

    def get_adjacency_matrices(self, triple_stats): # this assumes if there is a pathogen involved, there are both negative and positive edges for it!
        amp_target = pd.read_csv(self.positive_amp_target_path)
        neg_amp_target = pd.read_csv(self.negative_amp_target_path)

        amp_embeddings=os.listdir(self.amp_embedding_path)
        amp_embeddings=[x.split('.n')[0] for x in amp_embeddings]
        
        target_embeddings=os.listdir(self.target_embedding_path)
        target_embeddings=[x.split('.')[0] for x in target_embeddings]

        # since there might be some matchings in amp-target file that does not have an embedding for either an amp or a target
        # we need to eliminate the lines that suffers from either on of the conditions
        print(f"Number of AMPs in folder:{len(set(amp_embeddings))}")
        print(f"Number of AMPs in the positive link file:{len(set(amp_target['ID'].to_numpy()))}")
        print(f"Number of AMPs in the negative link file:{len(set(neg_amp_target['ID'].to_numpy()))}")
        all_pep =set(np.append(amp_target['ID'].to_numpy(), neg_amp_target['ID'].to_numpy()))
        print(f"Number of AMPs in total (unique): {len(np.unique(np.append(amp_target['ID'].to_numpy(), neg_amp_target['ID'].to_numpy())))}")

        amp_target = amp_target.loc[(amp_target["ID"].isin(amp_embeddings))]
        neg_amp_target = neg_amp_target.loc[(neg_amp_target["ID"].isin(amp_embeddings))]
        amp_target = amp_target.loc[amp_target["Pathogens"].isin(target_embeddings)] # amp_target = amp_target.loc[amp_target["Pathogens"].isin(target_embeddings.keys())]
        neg_amp_target = neg_amp_target.loc[neg_amp_target["Pathogens"].isin(target_embeddings)] # amp_target = amp_target.loc[amp_target["Pathogens"].isin(target_embeddings.keys())]

        # check vice versa as well
        amp_embeddings = {}
        tmp = np.unique(np.append(amp_target["ID"].to_numpy(), neg_amp_target["ID"].to_numpy()))
        print(f"Number of AMPs in the feature matrix:{len(tmp)}")
        print(f"Peptide embeddings cannot be found: {all_pep.difference(set(tmp))}")
        for amp in tmp:
            amp_embeddings[amp] = np.load(os.path.join(self.amp_embedding_path, f"{amp}.npy"))
        
        target_embeddings = {}
        tmp = np.unique(np.append(amp_target["Pathogens"].to_numpy(), neg_amp_target["Pathogens"].to_numpy()))
        print(f"Number of Targets in the feature matrix:{len(tmp)}")
        for target in tmp:
            target_embeddings[target] = np.load(os.path.join(self.target_embedding_path, f"{target}.npy"))
        
        # reaggregate to group based on AMP IDs
        # since we might have peptides that are not active against any pathogen, we need to add those as empty pathogen columns as well
        empty_rows = neg_amp_target[~neg_amp_target["ID"].isin(amp_target["ID"].to_numpy())].drop_duplicates( subset = ['Sequence',"ID"], keep = 'last').reset_index(drop = True) 
        empty_rows.loc[:,"Pathogens"] = "" # empty as they are negative edges!
        print(f"Empty rows len pos: {len(empty_rows)}")
        amp_target = pd.concat([amp_target, empty_rows]).reset_index(drop = True) 
        amp_target_tmp = pd.concat([amp_target, empty_rows]) # TODO PLEASE CHECK THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        amp_target = amp_target.groupby(["ID"]).agg({"Pathogens": lambda x: x.tolist()})
        amp_target_adj = pd.get_dummies(amp_target["Pathogens"].explode()).groupby(level=0).sum()
        
        # same thing applies for the negative edges
        empty_rows = amp_target_tmp[~amp_target_tmp["ID"].isin(neg_amp_target["ID"].to_numpy())].drop_duplicates( subset = ['Sequence',"ID"], keep = 'last').reset_index(drop = True) 
        empty_rows.loc[:,"Pathogens"] = "" # empty as they are negative edges!
        print(f"Empty rows len neg: {len(empty_rows)}")
        neg_amp_target = pd.concat([neg_amp_target,empty_rows])
        neg_amp_target = neg_amp_target.groupby(["ID"]).agg({"Pathogens": lambda x: x.tolist()})
        neg_amp_target_adj = pd.get_dummies(neg_amp_target["Pathogens"].explode()).groupby(level=0).sum()
        
        # AMP to Target or Target to AMP:
        amp_target_adj = amp_target_adj.reindex(columns=target_embeddings.keys(), index=amp_embeddings.keys())
        amp_target_adj = amp_target_adj.to_numpy()
        neg_amp_target_adj = neg_amp_target_adj.reindex(columns=target_embeddings.keys(), index=amp_embeddings.keys())
        neg_amp_target_adj = neg_amp_target_adj.to_numpy()

        # indices - consistent for amp & target pairs too
        amp_idx = {value: index for index, value in enumerate(amp_embeddings.keys())}
        target_idx = {value: index for index, value in enumerate(target_embeddings.keys())}
        
        print(f"Dimensions of AMP-Target Adjacency Matrix: {amp_target_adj.shape}")
        print(f"Dimensions of Negative AMP-Target Adjacency Matrix: {neg_amp_target_adj.shape}")
        print(f"Number of AMPs in the feature matrix:{len(amp_embeddings)}")
        print(f"Number of Targets in the feature matrix:{len(target_embeddings)}")
        
        if not triple_stats:
            amp_embeddings = [torch.mean(from_numpy(value), dim=0) for value in amp_embeddings.values()]
            target_embeddings = [torch.mean(from_numpy(value).view(-1, 512), dim=0) for value in target_embeddings.values()] # TODO automatic size instead of 512
        
        else:
            qs = torch.tensor([0.25, 0.5, 0.75])
            amp_embeddings = [torch.quantile(from_numpy(value), qs, dim=0).flatten() for value in amp_embeddings.values()]
            target_embeddings = [torch.quantile(from_numpy(value).view(-1, 512), qs, dim=0).flatten() for value in target_embeddings.values()]

        amp_embeddings = torch.stack(amp_embeddings, dim=0)
        amp_masks = torch.empty((amp_embeddings.size()[0], amp_embeddings.size()[0]))

        target_embeddings = torch.stack(target_embeddings, dim=0)
        target_masks = torch.empty((target_embeddings.size()[0], target_embeddings.size()[0]))
        
        del empty_rows
        return (csr_matrix(amp_target_adj), csr_matrix(neg_amp_target_adj), amp_idx, amp_embeddings, amp_masks, target_idx, target_embeddings, target_masks) 

    def build_graph_and_split_data(self, train_mes_passing, balance_negative):
        # since we are going to be classifying edges, we need to split the data based on the edges
        # we need to split the edges for the target-AMP pairs only! Also they need to be consistent with AMP-target and target-AMP links! 
        # we use target-target and peptide-peptide edges for message passing only!
        # now this uses the transductive scheme for link prediction described at: http://snap.stanford.edu/class/cs224w-2020/slides/08-GNN-application.pdf
        # positive graph
        utils.set_seed(constants.SEED)
        edge_cnt = len(self.amp_target_adj.nonzero()[0])
        val_cnt = int(edge_cnt * self.data_split[1])
        train_cnt = edge_cnt - val_cnt
        print(f'Edge count: {edge_cnt}')
        print(f'Validation count: {val_cnt}')
        if len(self.data_split) == 3:
            test_cnt = int(edge_cnt * self.data_split[2])
            train_cnt = train_cnt - test_cnt
        train_mp_cnt = int(train_mes_passing*train_cnt)
        
        # negative graph
        neg_edge_cnt = len(self.neg_amp_target_adj.nonzero()[0])
        neg_val_cnt = int(neg_edge_cnt * self.data_split[1])
        neg_train_cnt = neg_edge_cnt - neg_val_cnt
        print(f'Negative edge count: {neg_edge_cnt}')
        print(f'Negative validation count: {neg_val_cnt}')
        if len(self.data_split) == 3:
            neg_test_cnt = int(neg_edge_cnt * self.data_split[2])
            neg_train_cnt = neg_train_cnt - neg_test_cnt

        eids = np.arange(edge_cnt)
        eids = np.random.permutation(eids) # IDs are permuted here!

        neg_eids = np.arange(neg_edge_cnt)
        neg_eids = np.random.permutation(neg_eids) # negative IDs are permuted here!

        edges = self.amp_target_adj.nonzero()
        neg_edges = self.neg_amp_target_adj.nonzero()

        if len(self.data_split) == 3:
            test_u, test_v = edges[0][eids[:test_cnt]], edges[1][eids[:test_cnt]]
            test_mp_u, test_mp_v = edges[0][eids[test_cnt:]], edges[1][eids[test_cnt:]]
            neg_test_u, neg_test_v = neg_edges[0][neg_eids[:neg_test_cnt]], neg_edges[1][neg_eids[:neg_test_cnt]]
            
            val_u, val_v = edges[0][eids[test_cnt:test_cnt+val_cnt]], edges[1][eids[test_cnt:test_cnt+val_cnt]]
            val_mp_u, val_mp_v = edges[0][eids[test_cnt+val_cnt:]], edges[1][eids[test_cnt+val_cnt:]]
            neg_val_u, neg_val_v = neg_edges[0][neg_eids[neg_test_cnt:neg_test_cnt+neg_val_cnt]], neg_edges[1][neg_eids[neg_test_cnt:neg_test_cnt+neg_val_cnt]]
            
            train_u, train_v = edges[0][eids[test_cnt+val_cnt:-train_mp_cnt]], edges[1][eids[test_cnt+val_cnt:-train_mp_cnt]]
            train_mp_u, train_mp_v = edges[0][eids[-train_mp_cnt:]], edges[1][eids[-train_mp_cnt:]]
            neg_train_u, neg_train_v = neg_edges[0][neg_eids[neg_test_cnt+neg_val_cnt:]], neg_edges[1][neg_eids[neg_test_cnt+neg_val_cnt:]]
        else: 
            val_u, val_v = edges[0][eids[:val_cnt]], edges[1][eids[:val_cnt]]
            val_mp_u, val_mp_v = edges[0][eids[val_cnt:]], edges[1][eids[val_cnt:]]
            neg_val_u, neg_val_v = neg_edges[0][neg_eids[:neg_val_cnt]], neg_edges[1][neg_eids[:neg_val_cnt]]

            train_u, train_v = edges[0][eids[val_cnt:-train_mp_cnt]], edges[1][eids[val_cnt:-train_mp_cnt]]
            train_mp_u, train_mp_v = edges[0][eids[-train_mp_cnt:]], edges[1][eids[-train_mp_cnt:]]
            neg_train_u, neg_train_v = neg_edges[0][neg_eids[neg_val_cnt:]], neg_edges[1][neg_eids[neg_val_cnt:]]
        
        if balance_negative:
            neg_train_u = neg_train_u[:len(train_u)]
            neg_train_v = neg_train_v[:len(train_v)]

        num_nodes_dict = {'AMP': self.amp_embeddings.size()[0], 'Target': self.target_embeddings.size()[0]}
        train_graph = heterograph({
            ("AMP", "is_similar_p", "AMP"): ([0], [0]),
            ("Target", "is_similar_g", "Target"): ([0], [0]),
            ("AMP", "is_active", "Target"): (train_mp_u, train_mp_v),
            ("Target", "is_susceptable", "AMP"): (train_mp_v, train_mp_u) 
        }, num_nodes_dict=num_nodes_dict)
        train_graph_supervision = heterograph({
            ("AMP", "is_active", "Target"): (train_u, train_v)
        }, num_nodes_dict=num_nodes_dict)
        train_graph_neg = heterograph({
            ("AMP", "is_active", "Target"): (neg_train_u, neg_train_v)
        }, num_nodes_dict=num_nodes_dict)
     
        val_graph = heterograph({
            ("AMP", "is_similar_p", "AMP"): ([0], [0]),
            ("Target", "is_similar_g", "Target"): ([0], [0]),
            ("AMP", "is_active", "Target"): (val_mp_u, val_mp_v),
            ("Target", "is_susceptable", "AMP"): (val_mp_v, val_mp_u) 
        }, num_nodes_dict=num_nodes_dict)
        val_graph_supervision = heterograph({
            ("AMP", "is_active", "Target"): (val_u, val_v)
        }, num_nodes_dict=num_nodes_dict)

        val_graph_neg = heterograph({
            ("AMP", "is_active", "Target"): (neg_val_u, neg_val_v)
        }, num_nodes_dict=num_nodes_dict)

        if len(self.data_split) == 3:
            test_graph = heterograph({ 
                ("AMP", "is_similar_p", "AMP"): ([0], [0]),
                ("Target", "is_similar_g", "Target"): ([0], [0]),
                ("AMP", "is_active", "Target"): (test_mp_u, test_mp_v),
                ("Target", "is_susceptable", "AMP"): (test_mp_v, test_mp_u) 
            }, num_nodes_dict=num_nodes_dict)
            test_graph_supervision = heterograph({
            ("AMP", "is_active", "Target"): (test_u, test_v)
            }, num_nodes_dict=num_nodes_dict)   
        
            test_graph_neg = heterograph({
            ("AMP", "is_active", "Target"): (neg_test_u, neg_test_v)
            }, num_nodes_dict=num_nodes_dict)
            graphs = [train_graph, train_graph_supervision, train_graph_neg, val_graph, val_graph_supervision, val_graph_neg, test_graph, test_graph_supervision, test_graph_neg]
        else:
            graphs = [train_graph, train_graph_supervision, train_graph_neg, val_graph, val_graph_supervision, val_graph_neg]
        #del self.amp_target_adj
        #del self.neg_amp_target_adj
        #gc.collect()
        return graphs

