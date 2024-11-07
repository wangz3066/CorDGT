from cmath import isnan
import numpy as np
import torch
import time
from tqdm import tqdm

import networkx as nx

from utils import bisect_left_adapt

import hashlib

class MethodWLNodeColoring():
    data = None
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    def setting_init(self, node_list, link_list):
        for node in node_list:
            self.node_color_dict[node] = 1
            self.node_neighbor_dict[node] = {}

        for pair in link_list:
            u1, u2 = pair
            if u1 not in self.node_neighbor_dict:
                self.node_neighbor_dict[u1] = {}
            if u2 not in self.node_neighbor_dict:
                self.node_neighbor_dict[u2] = {}
            self.node_neighbor_dict[u1][u2] = 1
            self.node_neighbor_dict[u2][u1] = 1

    def WL_recursion(self, node_list):
        iteration_count = 1
        while True:
            new_color_dict = {}
            for node in node_list:
                neighbors = self.node_neighbor_dict[node]
                neighbor_color_list = [self.node_color_dict[neb] for neb in neighbors]
                color_string_list = [str(self.node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
                color_string = "_".join(color_string_list)
                hash_object = hashlib.md5(color_string.encode())
                hashing = hash_object.hexdigest()
                new_color_dict[node] = hashing
            color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
            for node in new_color_dict:
                new_color_dict[node] = color_index_dict[new_color_dict[node]]
            if self.node_color_dict == new_color_dict or iteration_count == self.max_iter:
                return
            else:
                self.node_color_dict = new_color_dict
            iteration_count += 1


    def run(self):
        node_list = self.data['idx']
        link_list = self.data['edges']
        self.setting_init(node_list, link_list)
        self.WL_recursion(node_list)
        return self.node_color_dict

    

class GraphMaker:
    def __init__(self, num_neigh, max_depth):
        self.num_neigh = num_neigh
        self.max_depth = max_depth

        total_node_num = 1
        tgt_node_num = 1
        self.offset_per_layer = [0]

        for i in range(max_depth):
            self.offset_per_layer.append(total_node_num)
            total_node_num += tgt_node_num * num_neigh[i]
            tgt_node_num *= num_neigh[i]

        self.offset_per_layer.append(total_node_num)
        self.total_node_num = total_node_num


    def get_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.eidx_g = np.zeros((self.batch_size, self.total_node_num, self.total_node_num))

    def update(self, neigh_eidx_l, depth):
        idx = []
        src_start = self.offset_per_layer[depth]
        src_end = self.offset_per_layer[depth+1]

        cur = src_end
        for i in range(src_start, src_end):
            for j in range(self.num_neigh[depth]):
                idx.append([i, cur+j])
            cur += self.num_neigh[depth]

        idx = np.array(idx)

        neigh_eidx = neigh_eidx_l.reshape(self.batch_size, -1)
        for i in range(idx.shape[0]):
            self.eidx_g[:, idx[i,0], idx[i,1]] = neigh_eidx[:, i]
            

class NeighborFinder:
    def __init__(self, adj_list, balance, seed, uniform=False, ):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """ 
       
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []

        for neighbors in adj_list:
        # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
        # We sort the list based on timestamp
            sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

        self.uniform = uniform
        self.random_state = np.random.RandomState(seed)

    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]
        
        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[2])
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            n_ts_l.extend([x[2] for x in curr])    
            
            off_set_l.append(len(n_idx_l))
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert(len(n_idx_l) == len(n_ts_l))
        assert(off_set_l[-1] == len(n_ts_l))
        
        return n_idx_l, n_ts_l, e_idx_l, off_set_l

    def find_before(self, src_idx, cut_time):
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

        Returns 3 lists: neighbors, edge_idxs, timestamps

        """

        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

        return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]



    def get_temporal_neighbor(self, src_idx_l, cut_time_l, depth, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert(len(src_idx_l) == len(cut_time_l))
        
        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        
        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts = self.find_before(src_idx, cut_time)


            if len(ngh_idx) > 0:
                if self.uniform:
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)
                    
                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]
                    
                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
                else:
                    ngh_ts = ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]
                    ngh_eidx = ngh_eidx[:num_neighbors]
                    
                    assert(len(ngh_idx) <= num_neighbors)
                    assert(len(ngh_ts) <= num_neighbors)
                    assert(len(ngh_eidx) <= num_neighbors)
                    
                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eidx_batch[i,  num_neighbors - len(ngh_eidx):] = ngh_eidx


        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch



    def find_k_hop(self, k, src_idx_l, cut_time_l, graph_maker, num_neighbors):
        """Sampling the k-hop sub graph in tree struture
        """
        if k == 0:
            return ([], [], [])
        
        graph_maker.get_batch_size(src_idx_l.shape[0])
        
        node_records = [np.expand_dims(src_idx_l,-1)]
        t_records = [np.expand_dims(cut_time_l,-1)]

        batch = len(src_idx_l)
        layer_i = 0
        x, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, layer_i, num_neighbors[layer_i])
        one_hop_node = x
        one_hop_t = z
        node_records.append(x)
        graph_maker.update(y, layer_i)
        t_records.append(z)

        for layer_i in range(1, k):
            ngh_node_est, ngh_t_est = node_records[-1], t_records[-1]
            ngh_node_est = ngh_node_est.flatten()
            ngh_t_est = ngh_t_est.flatten()
            out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.get_temporal_neighbor(ngh_node_est, ngh_t_est, layer_i, num_neighbors[layer_i])
            out_ngh_node_batch = out_ngh_node_batch.reshape(batch, -1)
            graph_maker.update(out_ngh_eidx_batch, layer_i)
            out_ngh_t_batch = out_ngh_t_batch.reshape(batch, -1)

            node_records.append(out_ngh_node_batch)
            t_records.append(out_ngh_t_batch)

        node_records = np.concatenate(node_records, axis=1)
        t_records = np.concatenate(t_records, axis=1)

        return one_hop_node, one_hop_t, node_records, t_records, graph_maker.eidx_g, graph_maker.offset_per_layer

    
    def compute_degs(self):
        '''
        (only for data analysis)
        Return node average degrees and a numpy array of all node degrees
        '''
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        degs = []
        for n_idx, ts in zip(node_idx_l, node_ts_l):
            deg = len(self.find_before(n_idx, ts, need_deg=False)[0])
            degs.append(deg)
        degs = np.array(degs)
        return degs.mean(), degs

    def compute_ratio(self):
        '''
        (only for data analysis)
        Return the ratio : # temporal nodes that neighbor that is 1-hop neighbor and 2-hop neighbor simutaneously / # temporal node
        '''
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        cnt = 0
        one_hop_time_diff_mean_l = []
        one_hop_time_diff_std_l = []
        two_hop_time_diff_l = []
        sample_ratio_l = []
        step = 0
        for n_idx, ts in tqdm(zip(node_idx_l, node_ts_l)):

            node_l ,_ , time_l = self.find_before(n_idx, ts, need_deg=False)

            if len(node_l)>0:
                t_diff_l = ts - time_l
                t_diff_max = t_diff_l.max()
                t_diff_l = t_diff_l / t_diff_max
                t_diff_l_mean = t_diff_l.mean()
                t_diff_l_std = t_diff_l.std()

            most_early_interact = {}
            for node, time in zip(node_l, time_l):
                if node in most_early_interact.keys():
                    continue
                most_early_interact[node]=time
            flg = False
            for node, time in zip(node_l, time_l):
                node_l_two_hop , e, time_l_two_hop = self.find_before(node, time, need_deg=False)
                for n, t in zip(node_l_two_hop, time_l_two_hop):
                    if n in node_l and most_early_interact[n] < time:
                        one_hop_time_diff_mean_l.append(t_diff_l_mean)
                        one_hop_time_diff_std_l.append(t_diff_l_std)
                        two_hop_diff = (ts - most_early_interact[n]) / t_diff_max
                        one_hop_diff = (ts - time) / t_diff_max
                        two_hop_time_diff_l.append(two_hop_diff)
                        sample_ratio_l.append(two_hop_diff / one_hop_diff)
                        cnt += 1
                        flg = True
                        break
                if flg:
                    break

            step += 1
            if step >= 1000000:
                break
        
        ratio = cnt / step
        one_hop_time_diff_mean = np.array(one_hop_time_diff_mean_l).mean()
        one_hop_time_diff_std = np.array(one_hop_time_diff_std_l).mean()
        two_hop_time_diff = np.array(two_hop_time_diff_l).mean()
        sample_ratio = np.array(sample_ratio_l).mean()

        return ratio, one_hop_time_diff_mean, one_hop_time_diff_std, two_hop_time_diff, sample_ratio


    def nodets2key(self, node: int, ts: float, depth:int):
        key = '-'.join([str(node), str(int(round(ts))), str(depth)])
        return key

    def query_neighbors(self, src_idx_l, cut_time_l, query_idx_l):
        is_neighbor = np.zeros_like(query_idx_l)
        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts = self.find_before(src_idx, cut_time)
            for j in range(len(query_idx_l[i])):
                if query_idx_l[i][j] in ngh_idx and query_idx_l[i][j] != 0 :
                    is_neighbor[i,j]=1

        return is_neighbor
        
  