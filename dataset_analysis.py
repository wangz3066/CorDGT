import pandas as pd
import numpy as np

def nums2str(n1, n2):
    return '_u_' + str(n1) + '_v_' + str(n2)

DATA = 'enron'
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))

src_l = g_df.u.values
dst_l = g_df.i.values
ts_l = g_df.ts.values

edge_dict = {}

for src, dst, ts in zip(src_l, dst_l, ts_l):
    if nums2str(src, dst) in edge_dict:
        edge_dict[nums2str(src, dst)].append((ts))
    else:
        edge_dict[nums2str(src, dst)] = [ts]

std_l = []
for key in edge_dict: 
    if len(edge_dict[key])>2:
        tl = np.array(edge_dict[key]) 
        tl2 = []
        for i in range(1, len(tl)):
            tl2.append(tl[i]-tl[i-1])
        tl2 = np.array(tl2)
        if np.all(tl2==0):
            continue
        tl2 = (tl2 - tl2.min()) / (tl2.max()-tl2.min()+1e-32)
        std_l.append(np.std(tl2))
        

print('interval_std:', np.mean(np.array(std_l)))

    






