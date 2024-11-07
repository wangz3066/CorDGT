import logging
from turtle import forward
import time

import numpy as np
import torch
import math
import copy

import torch.nn as nn
import torch.nn.functional as F

import networkx as nx

from graph import GraphMaker

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        #self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        #x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, bias, mask=None):

        attn1 = torch.bmm(q, k.transpose(1, 2))

        q2 = q.unsqueeze(-1).permute(0,1,3,2)
        k2 = bias.permute(0,1,3,2)

        attn2 = torch.matmul(q2, k2).squeeze(-2)
        
        attn = attn1 + attn2

        attn = attn / self.temperature

        # attn += bias

        if mask is not None:
            attn = attn.masked_fill(mask,1e-10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]

        attn_ret = attn

        attn = self.dropout(attn) # [n * b, l_v, d]
                
        emb1 = torch.bmm(attn, v)

        attn3 = attn.unsqueeze(-1).permute(0,1,3,2)
        emb2 = torch.matmul(attn3, bias).squeeze(-2)

        output = emb1 + emb2
        
        return output, attn_ret

class ScaledDotProductAttention2(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, bias, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_v, d]
                
        output = torch.bmm(attn, v)
        
        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, h, bias, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = h.size()

        residual = h

        h = self.layer_norm(h)

        q = self.w_qs(h).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(h).view(sz_b, len_q, n_head, d_k)
        v = self.w_vs(h).view(sz_b, len_q, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_v) # (n*b) x lv x dv


        if mask is not None:
            mask = mask.unsqueeze(0).repeat(self.n_head,1,1,1).view(-1, len_q, len_q)

        bias = bias.repeat(self.n_head,1,1,1).squeeze(-1)

        # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, bias, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)

        attn = attn.view(n_head, sz_b, len_q, len_q)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))

        output += residual
        
        return output, attn
    

class MapBasedMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)
        
        nn.init.xavier_normal_(self.fc.weight)
        
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)
        
        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)
        
        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        q = torch.unsqueeze(q, dim=2) # [(n*b), lq, 1, dk]
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3]) # [(n*b), lq, lk, dk]
        
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        k = torch.unsqueeze(k, dim=1) # [(n*b), 1, lk, dk]
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3]) # [(n*b), lq, lk, dk]
        
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        
        mask = mask.repeat(n_head, 1, 1) # (n*b) x lq x lk
        
        ## Map based Attention
        #output, attn = self.attention(q, k, v, mask=mask)
        q_k = torch.cat([q, k], dim=3) # [(n*b), lq, lk, dk * 2]
        attn = self.weight_map(q_k).squeeze(dim=3) # [(n*b), lq, lk]
        
        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_q, l_k]
        
        # [n * b, l_q, l_k] * [n * b, l_v, d_v] >> [n * b, l_q, d_v]
        output = torch.bmm(attn, v)
        
        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn
    
def expand_last_dim(x, num):
    view_size = list(x.size()) + [1]
    expand_size = list(x.size()) + [num]
    return x.view(view_size).expand(expand_size)

class FixedSinCosEncode(torch.nn.Module):
    '''
    Positional encoding used in "Attention Is All You Need."

    '''

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        num_timescales = hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            max(num_timescales - 1, 1))

        self.inv_timescales = min_timescale * np.exp(
            np.arange(num_timescales) *
            -log_timescale_increment)

    def forward(self, x):

        position = x

        scaled_time = position[:,:, np.newaxis] * self.inv_timescales[np.newaxis,:]
        scaled_time = np.concatenate([np.sin(scaled_time),np.cos(scaled_time)],axis=-1)

        signal = torch.from_numpy(scaled_time)

        return signal


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        
    def forward(self, ts, device):
        # ts: [N, L]
        batch_size = ts.shape[0]
        seq_len = ts.shape[1]

        ts_th = torch.tensor(ts).to(device)

        ts_th = ts_th.view(batch_size, seq_len, 1)# [N, L, 1]
        map_ts = ts_th * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)
        
        harmonic = torch.cos(map_ts)

        return harmonic #self.dense(harmonic)
    
    
class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()
        
        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)
        
    def forward(self, ts):
        # ts: [N, L]

        ts_emb = self.pos_embeddings(ts)
        return ts_emb
    

class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim
        
    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim
        
        self.att_dim = feat_dim + edge_dim + time_dim
        
        self.act = torch.nn.ReLU()
        
        self.lstm = torch.nn.LSTM(input_size=self.att_dim, 
                                  hidden_size=self.feat_dim, 
                                  num_layers=1, 
                                  batch_first=True)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        seq_x = torch.cat([seq, seq_e, seq_t], dim=2)
            
        _, (hn, _) = self.lstm(seq_x)
        
        hn = hn[-1, :, :] #hn.squeeze(dim=0)

        out = self.merger.forward(hn, src)
        return out, None
    

class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merger = MergeLayer(edge_dim + feat_dim, feat_dim, feat_dim, feat_dim)
        
    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        src_x = src
        seq_x = torch.cat([seq, seq_e], dim=2) #[B, N, De + D]
        hn = seq_x.mean(dim=1) #[B, De + D]
        output = self.merger(hn, src_x)
        return output, None
    

class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, edge_dim, time_dim, 
                 attn_mode='prod', n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()
        
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        
        self.edge_in_dim = (feat_dim + edge_dim + time_dim)
        self.model_dim = self.edge_in_dim

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        assert(self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode
        
        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out)
            self.logger.info('Using scaled prod attention')
            
        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head, 
                                             d_model=self.model_dim, 
                                             d_k=self.model_dim // n_head, 
                                             d_v=self.model_dim // n_head, 
                                             dropout=drop_out)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')
        
        
    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        src_ext = torch.unsqueeze(src, dim=1) # src [B, 1, D]
        src_e_ph = torch.zeros_like(src_ext)
        q = torch.cat([src_ext, src_e_ph, src_t], dim=2) # [B, 1, D + De + Dt] -> [B, 1, D]
        k = torch.cat([seq, seq_e, seq_t], dim=2) # [B, 1, D + De + Dt] -> [B, 1, D]
        
        mask = torch.unsqueeze(mask, dim=2) # mask [B, N, 1]
        mask = mask.permute([0, 2, 1]) #mask [B, 1, N]

        # # target-attention
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask) # output: [B, 1, D + Dt], attn: [B, 1, N]
        output = output.squeeze()
        attn = attn.squeeze()

        output = self.merger(output, src)
        return output, attn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.reset()
    
    def forward(self, x):
        x = self.fc2(torch.relu(self.fc1(x)))

        return x

    def reset(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super().__init__()

        self.MSA = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)

        self.layer_norm = nn.LayerNorm(d_v, eps=1e-6)

        self.fc1 = nn.Linear(d_model, 4*d_model)
        self.fc2 = nn.Linear(4*d_model,d_model)
        # self.activation = nn.ReLU()
        self.activation = nn.ReLU()

    def forward(self, emb_n, bias, mask):
        emb_n, attn = self.MSA(emb_n, bias, mask)  
        residual = emb_n

        emb_n = self.layer_norm(emb_n)

        emb_n = self.fc1(emb_n)
        emb_n = self.activation(emb_n)
        emb_n = self.fc2(emb_n)

        emb_n += residual

        return emb_n, attn


class Transformer(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, stpe_dim, d_model, n_head, num_layer, dropout):
        super().__init__()

        input_dim = node_feat_dim + stpe_dim

        self.fc_node_1 = nn.Linear(input_dim, d_model)

        self.num_layer = num_layer
        self.d_model = d_model

        self.fc_edge = nn.ModuleList([])
        for i in range(num_layer):
            self.fc_edge.append(nn.Linear(edge_feat_dim, d_model))

        self.layers = nn.ModuleList([])
        for i in range(num_layer):
            self.layers.append(TransformerBlock(d_model, n_head, d_model,d_model, dropout))

    def forward(self, node_feat, stpe_emb, edge_feat, mask_attn):
        '''
        node_feat : B * N * d_n 
        edge_feat : B * N * N * d_e
        '''

        emb_n = torch.cat([node_feat, stpe_emb], dim=2)

        emb_n = self.fc_node_1(emb_n)

        attn_l = []

        for i in range(self.num_layer):
            bias_e = self.fc_edge[i](edge_feat)
            bias = bias_e
            emb_n, attn = self.layers[i](emb_n, bias, mask_attn)
            attn_l.append(attn)
 
        output = emb_n.mean(dim=1)

        return output, attn_l
    
class Memory():
    def __init__(self, total_node_num) -> None:
        self.total_node_num = total_node_num
        self.history_interact_count, self.most_recent_interact = {}, {}

    def update(self, src_idx_l, tgt_idx_l, cur_time_l):
        for i in range(len(src_idx_l)):
            k1 = self.nums2str(src_idx_l[i], tgt_idx_l[i])
            k2 = self.nums2str(src_idx_l[i], tgt_idx_l[i])
            if k1 in self.history_interact_count:
                self.history_interact_count[k1] += 1
                self.history_interact_count[k2] += 1
                self.most_recent_interact[k1] = np.maximum(self.most_recent_interact[k1], cur_time_l[i])
                self.most_recent_interact[k2] = np.maximum(self.most_recent_interact[k2], cur_time_l[i])
            else:
                self.history_interact_count[k1] = 1
                self.history_interact_count[k2] = 1
                self.most_recent_interact[k1] = cur_time_l[i]
                self.most_recent_interact[k2] = cur_time_l[i]
            

    def nums2str(self, n1, n2):
        return '_u_' + str(n1) + '_v_' + str(n2)

    def reset(self):
        self.__init__(self.total_node_num)

    def backup_memory(self):
        return self.history_interact_count.copy(), self.most_recent_interact.copy()

    def restore_memory(self, backup):
        self.history_interact_count = backup[0].copy()
        self.most_recent_interact = backup[1].copy()

class TGAN(torch.nn.Module):
    def __init__(self, ngh_finder, n_feat, e_feat, stpe_dim, d_model, num_neigh, node_num, num_layers=3, n_head=2, drop_out=0.1, max_depth=2, alpha=10, eta=10000, beta=1):
        super(TGAN, self).__init__()
        
        self.num_layers = num_layers 
        self.ngh_finder = ngh_finder
        self.logger = logging.getLogger(__name__)

        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))

        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)

        self.node_feat_dim = self.n_feat_th.shape[1]
        self.edge_feat_dim = self.e_feat_th.shape[1]
        self.stpe_dim = stpe_dim
        
        self.total_node_num = node_num + 1
        self.max_depth = max_depth
        self.alpha = alpha
        self.eta = eta
        self.beta = beta

        self.memory = Memory(self.total_node_num)

        self.recent_encoder = FixedSinCosEncode(hidden_size=self.stpe_dim)

        # self.recent_encoder = TimeEncode(expand_dim=self.stpe_dim)
        
        self.f2 = MLP(input_dim=self.stpe_dim, hidden_dim=d_model, output_dim=self.stpe_dim)
        self.f3 = MLP(input_dim=self.stpe_dim, hidden_dim=d_model, output_dim=self.stpe_dim)

        self.transformer = Transformer(self.node_feat_dim, self.edge_feat_dim, 2*stpe_dim ,d_model, n_head, num_layers, drop_out)

        self.graph_maker = GraphMaker(num_neigh=num_neigh, max_depth=max_depth)
        
        self.affinity_score = MergeLayer(d_model, d_model, d_model, 1) # torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)

        self.time_count = {
            "sample":[],
            'temporal':[],
            'spatial':[],
            'forward':[]
        }
        
    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
         
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        
        return score

    def contrast(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l, device, num_neighbors, update_mem_with_neigh = False):
        pos_src_embed, attn_l = self.tem_conv(src_idx_l, cut_time_l, target_idx_l, self.num_layers, device, num_neighbors, update_mem_with_neigh = update_mem_with_neigh)
        pos_target_embed, _ = self.tem_conv(target_idx_l, cut_time_l, src_idx_l, self.num_layers, device, num_neighbors)
        neg_src_embed, _ = self.tem_conv(src_idx_l, cut_time_l, background_idx_l, self.num_layers, device, num_neighbors)
        neg_tgt_embed, _ = self.tem_conv(background_idx_l, cut_time_l, src_idx_l, self.num_layers, device, num_neighbors)
        pos_score = self.affinity_score(pos_src_embed, pos_target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(neg_src_embed, neg_tgt_embed).squeeze(dim=-1)
        self.memory.update(src_idx_l, target_idx_l, cut_time_l)
                                                                                                              
        return pos_score.sigmoid(), neg_score.sigmoid(), attn_l

    def tem_conv(self, src_idx_l, cut_time_l, tgt_idx_l, curr_layers, device, num_neighbors=20, update_mem_with_neigh = False):
        assert(curr_layers >= 0)
    
        batch_size = len(src_idx_l)

        src_one_hop_node, src_one_hop_t, src_neigh_node_records, src_neigh_t_records, src_neigh_eidx_g, src_neigh_offset_per_layer = self.ngh_finder.find_k_hop(self.max_depth, src_idx_l, cut_time_l, self.graph_maker, num_neighbors)
        tgt_one_hop_node, tgt_one_hop_t, tgt_neigh_node_records, tgt_neigh_t_records, _, _ = self.ngh_finder.find_k_hop(self.max_depth, tgt_idx_l, cut_time_l, self.graph_maker, num_neighbors)

        if update_mem_with_neigh: 
            self.update_mem_with_neigh(src_idx_l, src_one_hop_node, src_one_hop_t)
            self.update_mem_with_neigh(tgt_idx_l, tgt_one_hop_node, tgt_one_hop_t)
            
        src_neigh_node_batch = torch.from_numpy(src_neigh_node_records).long().to(device)
        src_neigh_time_batch = torch.from_numpy(src_neigh_t_records).float().to(device)
        
        eidx_g = torch.from_numpy(src_neigh_eidx_g).long().to(device)

        # temporal encoding
        neigh_to_src_t = self.recent_interact_src_to_neigh(src_idx_l, src_neigh_node_records[:,0:], cut_time_l)
        neigh_to_tgt_t = self.recent_interact_src_to_neigh(tgt_idx_l, src_neigh_node_records[:,0:], cut_time_l)

        neigh_to_src_time_encode = self.recent_encoder(neigh_to_src_t).float().to(device)
        neigh_to_tgt_time_encode = self.recent_encoder(neigh_to_tgt_t).float().to(device)

        # spatial encoding
        neigh_to_src_dis = self.get_spatial_distance(src_neigh_node_records, src_neigh_node_records, src_neigh_offset_per_layer)
        neigh_to_dst_dis = self.get_spatial_distance(src_neigh_node_records, tgt_neigh_node_records, src_neigh_offset_per_layer)

        neigh_to_src_space_encode = self.recent_encoder(neigh_to_src_dis).float().to(device)
        neigh_to_dst_space_encode = self.recent_encoder(neigh_to_dst_dis).float().to(device)

        neigh_to_src_stpe = torch.cat([self.f2(neigh_to_src_time_encode), self.f3(neigh_to_src_space_encode)], dim=-1)
        neigh_to_dst_stpe = torch.cat([self.f2(neigh_to_tgt_time_encode), self.f3(neigh_to_dst_space_encode)], dim=-1)

        neigh_stpe_b = neigh_to_src_stpe + neigh_to_dst_stpe

        # masking
        num_node = eidx_g.shape[1]
        neigh_to_src_dis = np.zeros((batch_size, num_node))
        offsets = src_neigh_offset_per_layer

        for i in range(1, self.max_depth):
            neigh_to_src_dis[:, offsets[i]:offsets[i+1]]= i
        neigh_to_src_dis[:,offsets[self.max_depth]:]= self.max_depth
        src_dis_th = torch.from_numpy(neigh_to_src_dis).long().to(device)
        
        mask_node = src_neigh_node_batch.unsqueeze(-1).repeat(1,1,num_node) == 0

        mask_l = src_dis_th.unsqueeze(-1) > src_dis_th.unsqueeze(1)

        mask_t = src_neigh_time_batch.unsqueeze(-1) < src_neigh_time_batch.unsqueeze(1)

        mask_attn = mask_t |  mask_node | mask_l

        mask_attn = torch.logical_or(torch.logical_or(mask_t, mask_node),mask_l)

        node_feat = self.node_raw_embed(src_neigh_node_batch)
        edge_emb = self.edge_raw_embed(eidx_g)

        emb, attn_l = self.transformer(node_feat, neigh_stpe_b, edge_emb, mask_attn)

        return emb, attn_l

    def recent_interact_src_to_neigh(self, src_idx_l, neigh_idx_l, cur_time_l):
        most_recent_interact = np.zeros_like(neigh_idx_l)
        interact_cnt = np.zeros_like(neigh_idx_l)
        final_dis = np.zeros_like(neigh_idx_l)


        for i in range(neigh_idx_l.shape[0]):
            for j in range(neigh_idx_l.shape[1]):
                if neigh_idx_l[i,j] == 0:
                    most_recent_interact[i, j] = 0.0
                    interact_cnt[i, j] = 0
                    continue
                ky = self.memory.nums2str(src_idx_l[i], neigh_idx_l[i,j])
                if ky in self.memory.history_interact_count:
                    most_recent_interact[i, j] = self.memory.most_recent_interact[ky]
                    interact_cnt[i, j] = self.memory.history_interact_count[ky]
                else:
                    most_recent_interact[i, j] = 0.0
                    interact_cnt[i, j] = 0

        intensity = most_recent_interact / (interact_cnt * cur_time_l[:, np.newaxis] + 1)
        
        recent = (cur_time_l[:, np.newaxis] - most_recent_interact) / (cur_time_l[:, np.newaxis] +1)

        final_dis = self.beta * intensity + self.alpha * recent

        final_dis = final_dis * self.eta
        
        return final_dis

    def reset_memory(self):
        self.memory.reset()

    def node_label(self, dist_to_0, dist_to_1):
        # an implementation of the proposed double-radius node labeling (DRNL)
        d = (dist_to_0 + dist_to_1).astype(int)
        d_over_2, d_mod_2 = np.divmod(d, 2)
        labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
        labels[np.isinf(labels)] = 0
        labels[labels > 1e6] = 0  # set inf labels to 0
        labels[labels < -1e6] = 0  # set -inf labels to 0
        return labels
    
    def get_spatial_distance(self, src_neigh, tgt_neigh, offset):
        # Create a boolean mask for the matching elements

        mask = (src_neigh[:,:,None] == tgt_neigh[:,None,:])

        min_indices = np.zeros_like(src_neigh)+10000

        for i in range(mask.shape[-1]):
            index = np.where(mask[:,:,i]==True)
            min_indices[index[0], index[1]] = np.minimum(min_indices[index[0], index[1]], i)

        transformed = np.zeros_like(src_neigh)
        for i in range(len(offset)-1):
            mask = np.logical_and(min_indices >= offset[i], min_indices < offset[i+1])
            transformed[mask] = i

        transformed[min_indices >= offset[-1]] = len(offset)
        transformed[src_neigh==0] = len(offset)
        
        transformed = 10000 * transformed

        return transformed
    
    def get_spatial_distance2(self, src_neigh, tgt_neigh, offset):
        # Create a boolean mask for the matching elements

        mask = (src_neigh[:,:,None] == tgt_neigh[:,None,:]).astype(float) # B*N*N
        bs, n = mask.shape[0], mask.shape[1]

        mask = mask.reshape(bs*n,-1)
        dis = np.zeros((bs*n,1)).astype(float)
        cnt = np.ones((bs*n,1)).astype(float)

        cnt[:,0] += np.sum(mask, axis=-1)
        offset_to_weight = np.zeros((offset[-1],1))
        for i in range(len(offset)-1):
            offset_to_weight[offset[i]:offset[i+1],:]=i

        for i in range(n):
            dis[:,0] += mask[:, i]*(len(offset)-1-offset_to_weight[i,0])

        dis = dis.reshape(bs,n)

        dis[src_neigh==0] == len(offset)

        dis = 10000 * dis

        return dis
    
    def update_mem_with_neigh(self, src_idx_l, neigh_idx_l, ts_l):
        for i in range(neigh_idx_l.shape[1]):
            self.memory.update(src_idx_l, neigh_idx_l[:, i], ts_l[:, i])


