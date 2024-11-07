"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
import argparse

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
#import numba

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from module import TGAN
from graph import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler, compute_time_statistics

### Argument and global variables
parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit', default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', nargs='*', default=['64', '1'], help='a list of neighbor sampling numbers for different hops, when only a single element is input n_layer will be activated')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
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
parser.add_argument('--beta',type=float, default=1.0, help='coefficient to balance intensity and recent in temporal distance')
parser.add_argument('--update_mem_with_neigh', action='store_true', help='whether to update memory with one hop neighbor')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = [int(n) for n in args.n_degree]
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
TIME_DIM = args.time_dim
STPE_DIM = args.stpe_dim
D_MODEL = args.d_model
MAX_DEPTH = args.max_depth
WEIGHT_DECAY = args.weight_decay
TIME = args.time
GAMMA = args.gamma
ALPHA = args.alpha
ETA = args.eta
BETA = args.beta

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'

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


def eval_one_epoch(hint, tgan, sampler, src, dst, ts, label, device, update_mem_with_neigh=False):
    val_ap, val_auc = [], [], 
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE=30
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]

            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(size)

            pos_prob, neg_prob, _ = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, device, NUM_NEIGHBORS, update_mem_with_neigh)
            
            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            
            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))
    return np.mean(val_ap), np.mean(val_auc)

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

mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])), int(0.1 * num_total_unique_nodes)))
mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)

valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)

train_src_l = src_l[valid_train_flag]
train_dst_l = dst_l[valid_train_flag]
train_ts_l = ts_l[valid_train_flag]
train_e_idx_l = e_idx_l[valid_train_flag]
train_label_l = label_l[valid_train_flag]

# define the new nodes sets for testing inductiveness of the model
train_node_set = set(train_src_l).union(train_dst_l)
assert(len(train_node_set - mask_node_set) == len(train_node_set))
new_node_set = total_node_set - train_node_set

# select validation and test dataset
valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
valid_test_flag = ts_l > test_time

is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(src_l, dst_l)])
nn_val_flag = valid_val_flag * is_new_node_edge
nn_test_flag = valid_test_flag * is_new_node_edge

# validation and test with all edges
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
# validation and test with edges that at least has one new node (not in training set)
nn_val_src_l = src_l[nn_val_flag]
nn_val_dst_l = dst_l[nn_val_flag]
nn_val_ts_l = ts_l[nn_val_flag]
nn_val_e_idx_l = e_idx_l[nn_val_flag]
nn_val_label_l = label_l[nn_val_flag]

nn_test_src_l = src_l[nn_test_flag]
nn_test_dst_l = dst_l[nn_test_flag]
nn_test_ts_l = ts_l[nn_test_flag]
nn_test_e_idx_l = e_idx_l[nn_test_flag]
nn_test_label_l = label_l[nn_test_flag]

## Initialize the data structure for graph and edge sampling
# build the graph for fast query
# graph only contains the training data (with 10% nodes removal)
adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, balance=1.0, seed=args.seed ,uniform=UNIFORM)

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, balance=1.0, seed=args.seed ,uniform=UNIFORM)


train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
val_rand_sampler = RandEdgeSampler(src_l, dst_l)
nn_val_rand_sampler = RandEdgeSampler(nn_val_src_l, nn_val_dst_l)
test_rand_sampler = RandEdgeSampler(src_l, dst_l)
nn_test_rand_sampler = RandEdgeSampler(nn_test_src_l, nn_test_dst_l)

### Model initialize
device = torch.device('cuda:{}'.format(GPU))
tgan = TGAN(train_ngh_finder, n_feat, e_feat, stpe_dim=STPE_DIM, d_model=D_MODEL,  num_neigh= NUM_NEIGHBORS, node_num=num_total_unique_nodes+1, num_layers=NUM_LAYER, n_head=NUM_HEADS, drop_out=DROP_OUT,max_depth=MAX_DEPTH, alpha=ALPHA, eta=ETA, beta=BETA)
optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.BCELoss()
tgan = tgan.to(device) 

num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)

logger.info('num of training instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)
np.random.shuffle(idx_list) 

early_stopper = EarlyStopMonitor()
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30],gamma=GAMMA)
total_time = 0.0
total_infer_time = 0.0
step = 0
for epoch in range(NUM_EPOCH):
    # Training 
    # training use only training graph

    step += 1

    tgan.reset_memory()

    tgan.ngh_finder = train_ngh_finder
    ap, auc, m_loss =  [], [], []
    np.random.shuffle(idx_list)
    logger.info('start {} epoch'.format(epoch))
    for k in tqdm(range(num_batch)):

        start_time = time.time()

        percent = 100 * k / num_batch
        if k % int(0.2 * num_batch) == 0:
            logger.info('progress: {0:10.4f}'.format(percent))

        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
        src_l_cut, dst_l_cut = train_src_l[s_idx:e_idx], train_dst_l[s_idx:e_idx]
        ts_l_cut = train_ts_l[s_idx:e_idx]
        label_l_cut = train_label_l[s_idx:e_idx]
        size = len(src_l_cut)
        src_l_fake, dst_l_fake = train_rand_sampler.sample(size)

        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)

        optimizer.zero_grad()
        tgan = tgan.train()
        pos_prob, neg_prob, attn_l = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, device, NUM_NEIGHBORS)


        loss = criterion(pos_prob, pos_label)
        loss += criterion(neg_prob, neg_label)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        end_time = time.time()
        total_time += end_time - start_time

    #   get training results
        with torch.no_grad():
            tgan = tgan.eval()
            pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            ap.append(average_precision_score(true_label, pred_score))
            # f1.append(f1_score(true_label, pred_label))
            m_loss.append(loss.item())
            auc.append(roc_auc_score(true_label, pred_score))
    
    # validation phase use all information

    tgan.ngh_finder = full_ngh_finder

    # Because the memory will be updated when transductive evaluation, we backup the training memory first for inductive evaluation.
    train_memory_backup = tgan.memory.backup_memory()

    start_eval = time.time()
    val_ap, val_auc = eval_one_epoch('val for old nodes', tgan, val_rand_sampler, val_src_l, 
    val_dst_l, val_ts_l, val_label_l, device)
    total_infer_time += time.time() - start_eval

    # backup the validation memory of transductive evaluation, which can be used for testing. 
    val_memory_backup = tgan.memory.backup_memory()

    # restore the training memory, begin inductive evaluation
    tgan.memory.restore_memory(train_memory_backup)

    nn_val_ap, nn_val_auc = eval_one_epoch('val for new nodes', tgan, nn_val_rand_sampler, nn_val_src_l, 
    nn_val_dst_l, nn_val_ts_l, nn_val_label_l, device, update_mem_with_neigh=args.update_mem_with_neigh)

    # restore the memory of transductive evaluation, which can be used for testing.
    tgan.memory.restore_memory(val_memory_backup)
        
    logger.info('epoch: {}:'.format(epoch))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info('train auc: {}, val auc: {}, new node val auc: {}'.format(np.mean(auc), val_auc, nn_val_auc))
    logger.info('train ap: {}, val ap: {}, new node val ap: {}'.format(np.mean(ap), val_ap, nn_val_ap))

    if early_stopper.early_stop_check(val_ap):
        logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
        best_model_path = get_checkpoint_path(early_stopper.best_epoch)
        tgan.load_state_dict(torch.load(best_model_path))
        logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        tgan.eval()
        break
    else:
        torch.save(tgan.state_dict(), get_checkpoint_path(epoch))


# testing phase use all information

# restore the validation memory which can be used for testing

print("***************OLD*************")


# Training has finished, we have loaded the best model, and we want to backup its current
# memory (which has seen validation edges) so that it can also be used when testing on unseen
# nodes
val_memory_backup = tgan.memory.backup_memory()

tgan.ngh_finder = full_ngh_finder
test_ap, test_auc = eval_one_epoch('test for old nodes', tgan, test_rand_sampler, test_src_l, 
test_dst_l, test_ts_l, test_label_l, device)

print("***************NEW*************")

tgan.memory.restore_memory(val_memory_backup)

nn_test_ap,  nn_test_auc = eval_one_epoch('test for new nodes', tgan, nn_test_rand_sampler, nn_test_src_l, 
nn_test_dst_l, nn_test_ts_l, nn_test_label_l, device, update_mem_with_neigh=args.update_mem_with_neigh)

logger.info('Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
logger.info('Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))

logger.info('Saving TGAN model')
torch.save(tgan.state_dict(), MODEL_SAVE_PATH)
logger.info('TGAN models saved')

print("Average training time per epoch: {}".format(total_time/step))
print("Average inference time per epoch: {}".format(total_infer_time/step))