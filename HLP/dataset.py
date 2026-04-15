import constants

import utils
import os

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from loguru import logger

from dgl import heterograph
from torch import from_numpy
import torch

class triAMPhTestData:
    def __init__(self, 
                positive_amp_target_path:str,
                negative_amp_target_path: str, 
                positive_test_path: str,
                negative_test_path: str,
                amp_embedding_path:str,
                target_embedding_path:str
                ):
        
        self.positive_amp_target_path = positive_amp_target_path
        self.negative_amp_target_path = negative_amp_target_path
        self.positive_test_path = positive_test_path
        self.negative_test_path = negative_test_path
        self.amp_embedding_path = amp_embedding_path
        self.target_embedding_path = target_embedding_path
        
        (self.message_passing_adj, self.test_adj, self.neg_test_adj, self.amp_idx, self.amp_embeddings,
            self.target_idx, self.target_embeddings) = self.get_adjacency_matrices()
        self.graphs = self.build_graph_and_split_data()
        
        logger.info(f'The number of unique peptides in the message passing set is {self.graphs[0].num_nodes('AMP')}')
        logger.info(f'The number of unique pathogens in the message passing set is {self.graphs[0].num_nodes('Target')}')
        logger.info(f'The number of peptide-pathogen pairs in the message passing set is {self.graphs[0].num_edges('is_active')}')
        
        logger.info(f'The number of unique peptides in the positive supervision set is {self.graphs[1].num_nodes('AMP')}')
        logger.info(f'The number of unique pathogens in the positive supervision set is {self.graphs[1].num_nodes('Target')}')
        logger.info(f'The number of peptide-pathogen pairs in the positive supervision set is {self.graphs[1].num_edges('is_active')}')
        
        logger.info(f'The number of unique peptides in the negative supervision set is {self.graphs[2].num_nodes('AMP')}')
        logger.info(f'The number of unique pathogens in the negative supervision set is {self.graphs[2].num_nodes('Target')}')
        logger.info(f'The number of peptide-pathogen pairs in the negative supervision set is {self.graphs[2].num_edges('is_active')}')

    def get_adjacency_matrices(self):
        # read data
        amp_target = pd.read_csv(self.positive_amp_target_path)
        neg_amp_target = pd.read_csv(self.negative_amp_target_path)
        amp_target = amp_target[["ID", "Pathogens"]]
        neg_amp_target = neg_amp_target[["ID", "Pathogens"]]
        amp_target["Merged"] = amp_target["ID"] + amp_target["Pathogens"]
        neg_amp_target["Merged"] = neg_amp_target["ID"] + neg_amp_target["Pathogens"]
        

        test = pd.read_csv(self.positive_test_path)
        neg_test = pd.read_csv(self.negative_test_path)
        test = test[["ID", "Pathogens"]]
        neg_test = neg_test[["ID", "Pathogens"]]
        test["Merged"] = test["ID"] + test["Pathogens"]
        neg_test["Merged"] = neg_test["ID"] + neg_test["Pathogens"]

        if len(test.loc[(test["Merged"].isin(amp_target["Merged"].to_numpy())) | (test["Merged"].isin(neg_amp_target["Merged"].to_numpy()))] | neg_test.loc[(neg_test["Merged"].isin(neg_amp_target["Merged"].to_numpy())) | (neg_test["Merged"].isin(amp_target["Merged"].to_numpy()))])>0:
            logger.warning('Your test files contain one or more peptide-pathogen pairs with the message passing files. They are being removed from the supervision graphs.')
            # ensure the test pairs are independent, the ones also occuring in the training set is removed
            test = test.loc[~(test["Merged"].isin(amp_target["Merged"].to_numpy())) & ~(test["Merged"].isin(neg_amp_target["Merged"].to_numpy()))]
            neg_test = neg_test.loc[~(neg_test["Merged"].isin(neg_amp_target["Merged"].to_numpy())) & ~(neg_test["Merged"].isin(amp_target["Merged"].to_numpy()))]
        
        if len(test) == 0:
            raise Exception("No positive independent test peptide-pathogen pairs!")
        elif len(neg_test) == 0:
            raise Exception("No negative independent test peptide-pathogen pairs!")
        elif len(test) == 0 and len(neg_test) == 0:
            raise Exception("No independent test peptide-pathogen pairs!")
        
        amp_embeddings=os.listdir(self.amp_embedding_path)
        amp_embeddings=[x.split('.n')[0] for x in amp_embeddings]
        
        target_embeddings=os.listdir(self.target_embedding_path)
        target_embeddings=[x.split('.n')[0] for x in target_embeddings]

        # since there might be some pairs in amp-target file that do not have an embedding for either an amp or a target
        amp_target["ID"] = amp_target["ID"].astype(str)
        amp_target = amp_target.loc[(amp_target["ID"].isin(amp_embeddings))]
        neg_amp_target["ID"] = neg_amp_target["ID"].astype(str)
        neg_amp_target = neg_amp_target.loc[(neg_amp_target["ID"].isin(amp_embeddings))]
        amp_target = amp_target.loc[amp_target["Pathogens"].isin(target_embeddings)]
        neg_amp_target = neg_amp_target.loc[neg_amp_target["Pathogens"].isin(target_embeddings)] 
        
        test["ID"] = test["ID"].astype(str)
        test = test.loc[(test["ID"].isin(amp_embeddings))]
        neg_test["ID"] = neg_test["ID"].astype(str)
        neg_test = neg_test.loc[(neg_test["ID"].isin(amp_embeddings))]
        test = test.loc[test["Pathogens"].isin(target_embeddings)] 
        neg_test = neg_test.loc[neg_test["Pathogens"].isin(target_embeddings)] 
        
        # ensure vice versa by only loading the embeddings that are present in the data files
        amp_embeddings = {}
        tmp = np.unique(np.concat([amp_target["ID"].to_numpy(), neg_amp_target["ID"].to_numpy(), test["ID"].to_numpy(), neg_test["ID"].to_numpy()]))
        for amp in tmp:
            amp_embeddings[amp] = np.load(os.path.join(self.amp_embedding_path, f"{amp}.npy"))
        
        target_embeddings = {}
        tmp = np.unique(np.concat([amp_target["Pathogens"].to_numpy(), neg_amp_target["Pathogens"].to_numpy(), test["Pathogens"].to_numpy(), neg_test["Pathogens"].to_numpy()]))
        for target in tmp:
            target_embeddings[target] = np.load(os.path.join(self.target_embedding_path, f"{target}.npy"))
        
        # Message Passing Adjacency Matrix 
        # since we might have peptides that are not active against any pathogen, we need to add those as empty pathogen columns
        empty_rows_mp = neg_amp_target[~neg_amp_target["ID"].isin(amp_target["ID"].to_numpy())].drop_duplicates(subset ='ID', keep = 'last').reset_index(drop = True) 
        empty_rows_mp["Pathogens"] = "" 
        # supervision peptide-pathogen pairs have to be annotated as empty and included in this dataframe too
        test_pep = pd.concat([test, neg_test])
        test_pep["Pathogens"] = ""
        test_pep = test_pep.drop_duplicates(subset="ID")
        
        message_passing_adj = pd.concat([amp_target, empty_rows_mp, test_pep]).reset_index(drop = True)[["ID", "Pathogens"]]
        message_passing_adj = message_passing_adj.groupby(["ID"]).agg({"Pathogens": lambda x: x.tolist()})
        message_passing_adj = pd.get_dummies(message_passing_adj["Pathogens"].explode()).groupby(level=0).sum()
        
        # Supervision Adjacency Matrix
        # since we might have peptides that are not active against any pathogen, we need to add those as empty pathogen columns
        empty_rows_test = neg_test[~neg_test["ID"].isin(test["ID"].to_numpy())].drop_duplicates( subset = "ID", keep = 'last').reset_index(drop = True) 
        empty_rows_test["Pathogens"] = "" # empty as they are negative edges!
        # message passing peptide-pathogen pairs have to be annotated as empty and should be included here for consistency
        mp = pd.concat([empty_rows_mp, amp_target])
        mp["Pathogens"] = ""
        mp = mp.drop_duplicates( subset = "ID", keep = 'last')
        test_adj = pd.concat([test, empty_rows_test, mp]).reset_index(drop = True)[["ID", "Pathogens"]]
        test_adj = test_adj.groupby(["ID"]).agg({"Pathogens": lambda x: x.tolist()})
        test_adj = pd.get_dummies(test_adj["Pathogens"].explode()).groupby(level=0).sum()
    
        #same for negative test
        empty_rows_neg = test[~test["ID"].isin(neg_test["ID"].to_numpy())].drop_duplicates( subset = "ID", keep = 'last').reset_index(drop = True) 
        empty_rows_neg["Pathogens"] = "" # empty as they are positive edges!
        # supervision peptide-pathogen pairs have to be annotated as empyt as well
        neg_test_adj = pd.concat([neg_test, empty_rows_neg, mp]).reset_index(drop = True)[["ID", "Pathogens"]]
        neg_test_adj = neg_test_adj.groupby(["ID"]).agg({"Pathogens": lambda x: x.tolist()})
        neg_test_adj = pd.get_dummies(neg_test_adj["Pathogens"].explode()).groupby(level=0).sum()
        
        # Arrange columns and rows so that we have consistant indices and IDs 
        message_passing_adj = message_passing_adj.reindex(columns=target_embeddings.keys(), index=amp_embeddings.keys()).fillna(0)
        message_passing_adj = message_passing_adj.to_numpy()
        test_adj = test_adj.reindex(columns=target_embeddings.keys(), index=amp_embeddings.keys()).fillna(0)
        test_adj = test_adj.to_numpy()
        neg_test_adj = neg_test_adj.reindex(columns=target_embeddings.keys(), index=amp_embeddings.keys()).fillna(0)
        neg_test_adj = neg_test_adj.to_numpy()
        
        # indices - consistent for amp & target pairs too
        amp_idx = {value: index for index, value in enumerate(amp_embeddings.keys())}
        target_idx = {value: index for index, value in enumerate(target_embeddings.keys())}
        amp_embeddings = [torch.mean(from_numpy(value), dim=0) for value in amp_embeddings.values()]
        target_embeddings = [torch.mean(from_numpy(value).view(-1, 512), dim=0) for value in target_embeddings.values()] # TODO automatic size instead of 512
        amp_embeddings = torch.stack(amp_embeddings, dim=0)
        target_embeddings = torch.stack(target_embeddings, dim=0)
        
        return (csr_matrix(message_passing_adj), csr_matrix(test_adj), csr_matrix(neg_test_adj), amp_idx, amp_embeddings, target_idx, target_embeddings) 

    def build_graph_and_split_data(self):
        """ generates the message passing, positive test and negative test graphs for the test data """
        utils.set_seed(constants.SEED)
        num_nodes_dict = {'AMP': self.amp_embeddings.size()[0], 'Target': self.target_embeddings.size()[0]}
        
        edges = self.message_passing_adj.nonzero()
        message_passing_graph = heterograph({
            ("AMP", "is_similar_p", "AMP"): ([0], [0]),
            ("Target", "is_similar_g", "Target"): ([0], [0]),
            ("AMP", "is_active", "Target"): (edges[0], edges[1]),
            ("Target", "is_susceptable", "AMP"): (edges[1], edges[0]) 
        }, num_nodes_dict=num_nodes_dict)
        
        edges = self.test_adj.nonzero()
        positive_graph = heterograph({
            ("AMP", "is_active", "Target"): (edges[0], edges[1]),
            ("Target", "is_susceptable", "AMP"): (edges[1], edges[0]) 
        }, num_nodes_dict=num_nodes_dict)
        
        edges = self.neg_test_adj.nonzero()
        negative_graph = heterograph({
            ("AMP", "is_active", "Target"): (edges[0], edges[1]),
            ("Target", "is_susceptable", "AMP"): (edges[1], edges[0]) 
        }, num_nodes_dict=num_nodes_dict)
    
        graphs = [message_passing_graph, positive_graph, negative_graph]

        return graphs

# ===========================================================================================================================
class triAMPhInductiveData:
    def __init__(self, 
                positive_amp_target_path:str,
                negative_amp_target_path: str, 
                amp_embedding_path:str,
                target_embedding_path:str,
                data_split: list,
                mes_passing:float = 0.5,
                balance_negative: bool = False
                ):
        
        self.positive_amp_target_path = positive_amp_target_path
        self.negative_amp_target_path = negative_amp_target_path
        self.amp_embedding_path = amp_embedding_path
        self.target_embedding_path = target_embedding_path
        self.data_split = data_split # [train_proportion, validation_proportion, test_proportion]
        
        (self.amp_target_adj, self.neg_amp_target_adj, self.amp_idx, self.amp_embeddings,
            self.target_idx, self.target_embeddings) = self.get_adjacency_matrices()
        self.graphs, self.amp_embeddings = self.build_graph_and_split_data(mes_passing, balance_negative)
        
        logger.info(f'The number of unique peptides in the training message passing set is {self.graphs[0].num_nodes('AMP')}')
        logger.info(f'The number of unique pathogens in the training message passing set is {self.graphs[0].num_nodes('Target')}')
        logger.info(f'The number of peptide-pathogen pairs in the training message passing set is {self.graphs[0].num_edges('is_active')}')
        logger.info(f'The number of unique peptides in the training positive supervision set is {self.graphs[1].num_nodes('AMP')}')
        logger.info(f'The number of unique pathogens in the training positive supervision set is {self.graphs[1].num_nodes('Target')}')
        logger.info(f'The number of peptide-pathogen pairs in the training positive supervision set is {self.graphs[1].num_edges('is_active')}')
        logger.info(f'The number of unique peptides in the training negative test set is {self.graphs[2].num_nodes('AMP')}')
        logger.info(f'The number of unique pathogens in the training negative test set is {self.graphs[2].num_nodes('Target')}')
        logger.info(f'The number of peptide-pathogen pairs in the training negative test set is {self.graphs[2].num_edges('is_active')}')

        logger.info(f'The number of unique peptides in the validation message passing set is {self.graphs[3].num_nodes('AMP')}')
        logger.info(f'The number of unique pathogens in the validation message passing set is {self.graphs[3].num_nodes('Target')}')
        logger.info(f'The number of peptide-pathogen pairs in the validation message passing set is {self.graphs[3].num_edges('is_active')}')
        logger.info(f'The number of unique peptides in the validation positive supervision set is {self.graphs[4].num_nodes('AMP')}')
        logger.info(f'The number of unique pathogens in the validation positive supervision set is {self.graphs[4].num_nodes('Target')}')
        logger.info(f'The number of peptide-pathogen pairs in the validation positive supervision set is {self.graphs[4].num_edges('is_active')}')
        logger.info(f'The number of unique peptides in the validation negative test set is {self.graphs[5].num_nodes('AMP')}')
        logger.info(f'The number of unique pathogens in the validation negative test set is {self.graphs[5].num_nodes('Target')}')
        logger.info(f'The number of peptide-pathogen pairs in the validation negative test set is {self.graphs[5].num_edges('is_active')}')
        
        
        if len(data_split) == 3:
            logger.info(f'The number of unique peptides in the test message passing set is {self.graphs[6].num_nodes('AMP')}')
            logger.info(f'The number of unique pathogens in the test message passing set is {self.graphs[6].num_nodes('Target')}')
            logger.info(f'The number of peptide-pathogen pairs in the test message passing set is {self.graphs[6].num_edges('is_active')}')
            logger.info(f'The number of unique peptides in the test positive supervision set is {self.graphs[7].num_nodes('AMP')}')
            logger.info(f'The number of unique pathogens in the test positive supervision set is {self.graphs[7].num_nodes('Target')}')
            logger.info(f'The number of peptide-pathogen pairs in the test positive supervision set is {self.graphs[7].num_edges('is_active')}')
            logger.info(f'The number of unique peptides in the test negative test set is {self.graphs[8].num_nodes('AMP')}')
            logger.info(f'The number of unique pathogens in the test negative test set is {self.graphs[8].num_nodes('Target')}')
            logger.info(f'The number of peptide-pathogen pairs in the test negative test set is {self.graphs[8].num_edges('is_active')}')

    def get_adjacency_matrices(self):
        amp_target = pd.read_csv(self.positive_amp_target_path)
        neg_amp_target = pd.read_csv(self.negative_amp_target_path)

        amp_embeddings=os.listdir(self.amp_embedding_path)
        amp_embeddings=[x.split('.n')[0] for x in amp_embeddings]
        
        target_embeddings=os.listdir(self.target_embedding_path)
        target_embeddings=[x.split('.n')[0] for x in target_embeddings]

        # since there might be some matchings in amp-target file that does not have an embedding for either an amp or a target
        amp_target = amp_target.loc[amp_target["ID"].isin(amp_embeddings)]
        neg_amp_target = neg_amp_target.loc[neg_amp_target["ID"].isin(amp_embeddings)]
        amp_target = amp_target.loc[amp_target["Pathogens"].isin(target_embeddings)] # amp_target = amp_target.loc[amp_target["Pathogens"].isin(target_embeddings.keys())]
        neg_amp_target = neg_amp_target.loc[neg_amp_target["Pathogens"].isin(target_embeddings)] # amp_target = amp_target.loc[amp_target["Pathogens"].isin(target_embeddings.keys())]
        
        # check vice versa as well
        amp_embeddings = {}
        tmp = np.unique(np.append(amp_target["ID"].to_numpy(), neg_amp_target["ID"].to_numpy()))
        for amp in tmp:
            amp_embeddings[amp] = np.load(os.path.join(self.amp_embedding_path, f"{amp}.npy"))
        
        target_embeddings = {}
        tmp = np.unique(np.append(amp_target["Pathogens"].to_numpy(), neg_amp_target["Pathogens"].to_numpy()))
        for target in tmp:
            target_embeddings[target] = np.load(os.path.join(self.target_embedding_path, f"{target}.npy"))
        
        # reaggregate to group based on AMP IDs
        # since we might have peptides that are not active against any pathogen, we need to add those as empty pathogen columns as well
        empty_rows = neg_amp_target[~neg_amp_target["ID"].isin(amp_target["ID"].to_numpy())].drop_duplicates( subset = "ID", keep = 'last').reset_index(drop = True) 
        empty_rows.loc[:,"Pathogens"] = "" # empty as they are negative edges!
        amp_target = pd.concat([amp_target, empty_rows]).reset_index(drop = True) 
        amp_target_tmp = amp_target.copy()
        amp_target = amp_target.groupby(["ID"]).agg({"Pathogens": lambda x: x.tolist()})
        amp_target_adj = pd.get_dummies(amp_target["Pathogens"].explode()).groupby(level=0).sum()
        
        # same thing applies for the negative edges
        empty_rows = amp_target_tmp[~amp_target_tmp["ID"].isin(neg_amp_target["ID"].to_numpy())].drop_duplicates( subset = "ID", keep = 'last').reset_index(drop = True) 
        empty_rows.loc[:,"Pathogens"] = "" # empty as they are negative edges!
        neg_amp_target = pd.concat([neg_amp_target,empty_rows])
        neg_amp_target = neg_amp_target.groupby(["ID"]).agg({"Pathogens": lambda x: x.tolist()})
        neg_amp_target_adj = pd.get_dummies(neg_amp_target["Pathogens"].explode()).groupby(level=0).sum()
        
        # AMP to Target or Target to AMP:
        amp_target_adj = amp_target_adj.reindex(columns=target_embeddings.keys(), index=amp_embeddings.keys()).fillna(0)
        amp_target_adj = amp_target_adj.to_numpy()
        neg_amp_target_adj = neg_amp_target_adj.reindex(columns=target_embeddings.keys(), index=amp_embeddings.keys()).fillna(0)
        neg_amp_target_adj = neg_amp_target_adj.to_numpy()
        
        # indices - consistent for amp & target pairs too
        amp_idx = {value: index for index, value in enumerate(amp_embeddings.keys())}
        self.amp_converter = {index: value for index, value in enumerate(amp_embeddings.keys())}
        target_idx = {value: index for index, value in enumerate(target_embeddings.keys())}
        
        amp_embeddings = [torch.mean(from_numpy(value), dim=0) for value in amp_embeddings.values()]
        target_embeddings = [torch.mean(from_numpy(value).view(-1, 512), dim=0) for value in target_embeddings.values()] # TODO automatic size instead of 512
            
        amp_embeddings = torch.stack(amp_embeddings, dim=0)
        target_embeddings = torch.stack(target_embeddings, dim=0)
        
        return (csr_matrix(amp_target_adj), csr_matrix(neg_amp_target_adj), amp_idx, amp_embeddings, target_idx, target_embeddings) 

    def build_graph_and_split_data(self, mes_passing, balance_negative):
        # here instead of splitting edges, we split nodes --> inductive learning, but only peptide nodes
        utils.set_seed(constants.SEED)
        node_cnt = self.amp_target_adj.shape[0]
        val_cnt = int(node_cnt * self.data_split[1])
        train_cnt = node_cnt - val_cnt
        if len(self.data_split) == 3:
            test_cnt = int(node_cnt * self.data_split[2])
            train_cnt = train_cnt - test_cnt
        
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

        num_nodes_dict = {'AMP': len(set(train_nodes)), 'Target': max(set(edges[1])|set(neg_edges[1]))+1}
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

        num_nodes_dict = {'AMP': len(set(val_nodes)), 'Target':  max(set(edges[1])|set(neg_edges[1]))+1}
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

            num_nodes_dict = {'AMP': len(set(test_nodes)), 'Target':  max(set(edges[1])|set(neg_edges[1]))+1}
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
            amp_embs = [self.amp_embeddings[train_nodes,:], self.amp_embeddings[val_nodes,:], self.amp_embeddings[test_nodes,:]]
        else:
            graphs = [train_graph, train_graph_supervision, train_graph_neg, val_graph, val_graph_supervision, val_graph_neg]
            amp_embs = [self.amp_embeddings[train_nodes,:], self.amp_embeddings[val_nodes,:]]
        return graphs, amp_embs
    
# ===========================================================================================================================
class triAMPhTransductiveData:
    def __init__(self, 
                positive_amp_target_path:str,
                negative_amp_target_path: str, 
                amp_embedding_path:str,
                target_embedding_path:str,
                data_split: list,
                train_mes_passing:float = 0.5,
                balance_negative:bool = False):
        
        self.positive_amp_target_path = positive_amp_target_path
        self.negative_amp_target_path = negative_amp_target_path
        self.amp_embedding_path = amp_embedding_path
        self.target_embedding_path = target_embedding_path
        self.data_split = data_split # [train_proportion, validation_proportion, test_proportion]
        
        (self.amp_target_adj, self.neg_amp_target_adj, self.amp_idx, self.amp_embeddings,
            self.target_idx, self.target_embeddings) = self.get_adjacency_matrices()
        self.graphs = self.build_graph_and_split_data(train_mes_passing, balance_negative)
        
        logger.info(f'The number of unique peptides in the training message passing set is {self.graphs[0].num_nodes('AMP')}')
        logger.info(f'The number of unique pathogens in the training message passing set is {self.graphs[0].num_nodes('Target')}')
        logger.info(f'The number of peptide-pathogen pairs in the training message passing set is {self.graphs[0].num_edges('is_active')}')
        logger.info(f'The number of unique peptides in the training positive supervision set is {self.graphs[1].num_nodes('AMP')}')
        logger.info(f'The number of unique pathogens in the training positive supervision set is {self.graphs[1].num_nodes('Target')}')
        logger.info(f'The number of peptide-pathogen pairs in the training positive supervision set is {self.graphs[1].num_edges('is_active')}')
        logger.info(f'The number of unique peptides in the training negative test set is {self.graphs[2].num_nodes('AMP')}')
        logger.info(f'The number of unique pathogens in the training negative test set is {self.graphs[2].num_nodes('Target')}')
        logger.info(f'The number of peptide-pathogen pairs in the training negative test set is {self.graphs[2].num_edges('is_active')}')

        logger.info(f'The number of unique peptides in the validation message passing set is {self.graphs[3].num_nodes('AMP')}')
        logger.info(f'The number of unique pathogens in the validation message passing set is {self.graphs[3].num_nodes('Target')}')
        logger.info(f'The number of peptide-pathogen pairs in the validation message passing set is {self.graphs[3].num_edges('is_active')}')
        logger.info(f'The number of unique peptides in the validation positive supervision set is {self.graphs[4].num_nodes('AMP')}')
        logger.info(f'The number of unique pathogens in the validation positive supervision set is {self.graphs[4].num_nodes('Target')}')
        logger.info(f'The number of peptide-pathogen pairs in the validation positive supervision set is {self.graphs[4].num_edges('is_active')}')
        logger.info(f'The number of unique peptides in the validation negative test set is {self.graphs[5].num_nodes('AMP')}')
        logger.info(f'The number of unique pathogens in the validation negative test set is {self.graphs[5].num_nodes('Target')}')
        logger.info(f'The number of peptide-pathogen pairs in the validation negative test set is {self.graphs[5].num_edges('is_active')}')
        
        
        if len(data_split) == 3:
            logger.info(f'The number of unique peptides in the test message passing set is {self.graphs[6].num_nodes('AMP')}')
            logger.info(f'The number of unique pathogens in the test message passing set is {self.graphs[6].num_nodes('Target')}')
            logger.info(f'The number of peptide-pathogen pairs in the test message passing set is {self.graphs[6].num_edges('is_active')}')
            logger.info(f'The number of unique peptides in the test positive supervision set is {self.graphs[7].num_nodes('AMP')}')
            logger.info(f'The number of unique pathogens in the test positive supervision set is {self.graphs[7].num_nodes('Target')}')
            logger.info(f'The number of peptide-pathogen pairs in the test positive supervision set is {self.graphs[7].num_edges('is_active')}')
            logger.info(f'The number of unique peptides in the test negative test set is {self.graphs[8].num_nodes('AMP')}')
            logger.info(f'The number of unique pathogens in the test negative test set is {self.graphs[8].num_nodes('Target')}')
            logger.info(f'The number of peptide-pathogen pairs in the test negative test set is {self.graphs[8].num_edges('is_active')}')
            
    def get_adjacency_matrices(self): # this assumes if there is a pathogen involved, there are both negative and positive edges for it!
        amp_target = pd.read_csv(self.positive_amp_target_path)
        neg_amp_target = pd.read_csv(self.negative_amp_target_path)

        amp_embeddings=os.listdir(self.amp_embedding_path)
        amp_embeddings=[x.split('.n')[0] for x in amp_embeddings]
        
        target_embeddings=os.listdir(self.target_embedding_path)
        target_embeddings=[x.split('.')[0] for x in target_embeddings]

        # since there might be some matchings in amp-target file that does not have an embedding for either an amp or a target
        all_pep =set(np.append(amp_target['ID'].to_numpy(), neg_amp_target['ID'].to_numpy()))
        amp_target = amp_target.loc[(amp_target["ID"].isin(amp_embeddings))]
        neg_amp_target = neg_amp_target.loc[(neg_amp_target["ID"].isin(amp_embeddings))]
        amp_target = amp_target.loc[amp_target["Pathogens"].isin(target_embeddings)] # amp_target = amp_target.loc[amp_target["Pathogens"].isin(target_embeddings.keys())]
        neg_amp_target = neg_amp_target.loc[neg_amp_target["Pathogens"].isin(target_embeddings)] # amp_target = amp_target.loc[amp_target["Pathogens"].isin(target_embeddings.keys())]

        # check vice versa as well
        amp_embeddings = {}
        tmp = np.unique(np.append(amp_target["ID"].to_numpy(), neg_amp_target["ID"].to_numpy()))
        for amp in tmp:
            amp_embeddings[amp] = np.load(os.path.join(self.amp_embedding_path, f"{amp}.npy"))
        
        target_embeddings = {}
        tmp = np.unique(np.append(amp_target["Pathogens"].to_numpy(), neg_amp_target["Pathogens"].to_numpy()))
        for target in tmp:
            target_embeddings[target] = np.load(os.path.join(self.target_embedding_path, f"{target}.npy"))
        
        # reaggregate to group based on AMP IDs
        # since we might have peptides that are not active against any pathogen, we need to add those as empty pathogen columns as well
        empty_rows = neg_amp_target[~neg_amp_target["ID"].isin(amp_target["ID"].to_numpy())].drop_duplicates( subset = ["ID"], keep = 'last').reset_index(drop = True) 
        empty_rows.loc[:,"Pathogens"] = "" # empty as they are negative edges!
        amp_target_tmp = amp_target.copy()
        amp_target = pd.concat([amp_target, empty_rows]).reset_index(drop = True) 
        amp_target = amp_target.groupby(["ID"]).agg({"Pathogens": lambda x: x.tolist()})
        amp_target_adj = pd.get_dummies(amp_target["Pathogens"].explode()).groupby(level=0).sum()
        
        # same thing applies for the negative edges
        empty_rows = amp_target_tmp[~amp_target_tmp["ID"].isin(neg_amp_target["ID"].to_numpy())].drop_duplicates( subset = ["ID"], keep = 'last').reset_index(drop = True) 
        empty_rows.loc[:,"Pathogens"] = "" # empty as they are negative edges!
        neg_amp_target = pd.concat([neg_amp_target,empty_rows])
        neg_amp_target = neg_amp_target.groupby(["ID"]).agg({"Pathogens": lambda x: x.tolist()})
        neg_amp_target_adj = pd.get_dummies(neg_amp_target["Pathogens"].explode()).groupby(level=0).sum()
        
        # AMP to Target or Target to AMP:
        amp_target_adj = amp_target_adj.reindex(columns=target_embeddings.keys(), index=amp_embeddings.keys()).fillna(0)
        amp_target_adj = amp_target_adj.to_numpy()
        neg_amp_target_adj = neg_amp_target_adj.reindex(columns=target_embeddings.keys(), index=amp_embeddings.keys()).fillna(0)
        neg_amp_target_adj = neg_amp_target_adj.to_numpy()

        # indices - consistent for amp & target pairs too
        amp_idx = {value: index for index, value in enumerate(amp_embeddings.keys())}
        target_idx = {value: index for index, value in enumerate(target_embeddings.keys())}
        amp_embeddings = [torch.mean(from_numpy(value), dim=0) for value in amp_embeddings.values()]
        target_embeddings = [torch.mean(from_numpy(value).view(-1, 512), dim=0) for value in target_embeddings.values()] # TODO automatic size instead of 512
        amp_embeddings = torch.stack(amp_embeddings, dim=0)
        target_embeddings = torch.stack(target_embeddings, dim=0)

        del empty_rows
        return (csr_matrix(amp_target_adj), csr_matrix(neg_amp_target_adj), amp_idx, amp_embeddings, target_idx, target_embeddings) 

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
        if len(self.data_split) == 3:
            test_cnt = int(edge_cnt * self.data_split[2])
            train_cnt = train_cnt - test_cnt
        train_mp_cnt = int(train_mes_passing*train_cnt)
        
        # negative graph
        neg_edge_cnt = len(self.neg_amp_target_adj.nonzero()[0])
        neg_val_cnt = int(neg_edge_cnt * self.data_split[1])
        neg_train_cnt = neg_edge_cnt - neg_val_cnt
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
        return graphs


