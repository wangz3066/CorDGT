"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import sys
import random
import argparse
import copy

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from utils import EarlyStopMonitor

from module import TGAN
from graph import NeighborFinder


class LR(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)




### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', nargs='*', default=['64', '1'], help='a list of neighbor sampling numbers for different hops, when only a single element is input n_layer will be activated')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--weight_decay',type=float, default=0, help='weight decay rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
parser.add_argument('--stpe_dim', type=int, default=100, help='Dimentions of the stpe embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimentions of the time embedding')
parser.add_argument('--d_model',type=int, default=64, help='hidden dimension of transformer.')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod', help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--seed',default=None, type=int, help='Random seed')
parser.add_argument('--max_depth',type=int, default=2, help='neighbor searching depth')
parser.add_argument('--gamma',type=float, default=0.25, help='LR reducing rate.')
parser.add_argument('--alpha',type=float, default=10, help='coefficient to balance intensity and recent in temporal distance')
parser.add_argument('--eta',type=float, default=10000, help='coefficient used in sincos PE.' )
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = [int(n) for n in args.n_degree]
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
TIME_DIM = args.time_dim
STPE_DIM = args.stpe_dim
D_MODEL = args.d_model
MAX_DEPTH = args.max_depth
TIME = args.time
ALPHA = args.alpha
ETA = args.eta

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Load data and train val test split
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))

val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values

max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())

max_time_dif = ts_l.max() - ts_l.min()

total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
num_total_unique_nodes = len(total_node_set)

valid_train_flag = (ts_l <= val_time)  
valid_test_flag = ts_l > test_time
valid_val_flag = np.logical_and(ts_l>val_time, ts_l<=test_time) 

train_src_l = src_l[valid_train_flag]
train_dst_l = dst_l[valid_train_flag]
train_ts_l = ts_l[valid_train_flag]
train_e_idx_l = e_idx_l[valid_train_flag]
train_label_l = label_l[valid_train_flag]

val_src_l = src_l[valid_val_flag]
val_dst_l = dst_l[valid_val_flag]
val_ts_l = ts_l[valid_val_flag]
val_e_idx_l = e_idx_l[valid_val_flag]
val_label_l = label_l[valid_val_flag]

test_src_l = src_l[valid_test_flag]
test_dst_l = dst_l[valid_test_flag]
test_ts_l = ts_l[valid_test_flag]
test_e_idx_l = e_idx_l[valid_test_flag]
test_label_l = label_l[valid_test_flag]

### Initialize the data structure for graph and edge sampling
adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, balance=1.0, uniform=UNIFORM)

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, balance=1.0, uniform=UNIFORM)

### Model initialize
device = torch.device('cuda:{}'.format(GPU))
tgan = TGAN(train_ngh_finder, n_feat, e_feat, stpe_dim=STPE_DIM, d_model=D_MODEL,  num_neigh= NUM_NEIGHBORS, node_num=num_total_unique_nodes+1, num_layers=NUM_LAYER, n_head=NUM_HEADS, drop_out=DROP_OUT,max_depth=MAX_DEPTH, alpha=ALPHA, eta=ETA)
# optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
# criterion = torch.nn.BCELoss()
tgan = tgan.to(device)

num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)
logger.debug('num of training instances: {}'.format(num_instance))
logger.debug('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)
np.random.shuffle(idx_list) 

logger.info('loading saved TGAN model')
model_path = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{DATA}.pth'
tgan.load_state_dict(torch.load(model_path))
tgan.eval()
logger.info('TGAN models loaded')
logger.info('Start training node classification task')

lr_model = LR(D_MODEL, drop=DROP_OUT)
lr_optimizer = torch.optim.Adam(lr_model.parameters(), lr=args.lr)
lr_model = lr_model.to(device)
tgan.ngh_finder = full_ngh_finder
idx_list = np.arange(len(train_src_l))
lr_criterion = torch.nn.BCELoss()
lr_criterion_eval = torch.nn.BCELoss()

def eval_epoch(src_l, dst_l, ts_l, label_l, batch_size, lr_model, tgan, num_layer=NODE_LAYER):
    pred_prob = np.zeros(len(src_l))
    loss = 0
    num_instance = len(src_l)
    num_batch = math.ceil(num_instance / batch_size)
    with torch.no_grad():
        lr_model.eval()
        tgan.eval()
        for k in range(num_batch):          
            s_idx = k * batch_size
            e_idx = min(num_instance - 1, s_idx + batch_size)
            src_l_cut = src_l[s_idx:e_idx]
            dst_l_cut = dst_l[s_idx:e_idx]
            ts_l_cut = ts_l[s_idx:e_idx]
            label_l_cut = label_l[s_idx:e_idx]
            size = len(src_l_cut)
            src_embed, _ = tgan.tem_conv(src_l_cut, ts_l_cut, dst_l_cut, NODE_LAYER, device=device, num_neighbors=NUM_NEIGHBORS)
            tgan.memory.update(src_l_cut, dst_l_cut, ts_l_cut)           
            src_label = torch.from_numpy(label_l_cut).float().to(device)
            lr_prob = lr_model(src_embed).sigmoid()
            loss += lr_criterion_eval(lr_prob, src_label).item()
            pred_prob[s_idx:e_idx] = lr_prob.cpu().numpy()

    auc_roc = roc_auc_score(label_l, pred_prob)
    return auc_roc, loss / num_instance


best_test_auc = 0.0
best_val_auc = 0.0
early_stopper = EarlyStopMonitor(max_round=args.patience)
for epoch in range(args.n_epoch):
    lr_pred_prob = np.zeros(len(train_src_l))
    np.random.shuffle(idx_list)
    tgan = tgan.eval()
    lr_model = lr_model.train()

    tgan.reset_memory()

    #num_batch
    for k in tqdm(range(num_batch)):
        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
        src_l_cut = train_src_l[s_idx:e_idx]
        dst_l_cut = train_dst_l[s_idx:e_idx]
        ts_l_cut = train_ts_l[s_idx:e_idx]
        label_l_cut = train_label_l[s_idx:e_idx]
        
        size = len(src_l_cut)
        
        lr_optimizer.zero_grad()
        with torch.no_grad():
            src_embed, _ = tgan.tem_conv(src_l_cut, ts_l_cut, dst_l_cut, NODE_LAYER, device=device, num_neighbors=NUM_NEIGHBORS)
            tgan.memory.update(src_l_cut, dst_l_cut, ts_l_cut)
        
        src_label = torch.from_numpy(label_l_cut).float().to(device)
        lr_prob = lr_model(src_embed).sigmoid()
        lr_loss = lr_criterion(lr_prob, src_label)
        lr_loss.backward()
        lr_optimizer.step()

    val_auc, val_loss = eval_epoch(val_src_l, val_dst_l, val_ts_l, val_label_l, BATCH_SIZE, lr_model, tgan)
    if val_auc > best_val_auc:
        best_lr_model = copy.deepcopy(lr_model)

    #torch.save(lr_model.state_dict(), './saved_models/edge_{}_wkiki_node_class.pth'.format(DATA))
    logger.info(f'epoch {epoch} | train loss: {lr_loss}| val loss: {val_loss}| val auc: {val_auc}')
    best_val_auc = max(best_val_auc, val_auc)

test_auc, test_loss = eval_epoch(test_src_l, test_dst_l, test_ts_l, test_label_l, BATCH_SIZE, best_lr_model, tgan)
#torch.save(lr_model.state_dict(), './saved_models/edge_{}_wkiki_node_class.pth'.format(DATA))
logger.info(f'test auc: {test_auc}')




 




