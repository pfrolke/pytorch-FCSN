import torch

import numpy as np
from scipy import interpolate

from knapsack_solver import knapsack

from scipy.stats import kendalltau, spearmanr
from scipy.stats import rankdata

from sklearn.metrics import precision_recall_fscore_support

import sys

def eval_metrics(y_pred, y_true):
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return precision, recall, fscore


def select_keyshots(video_info, pred_score, inter_method = "upsample"):
    """
    input:
        video_info: specific video of *.h5 file
        pred_score: [320] key frame score in every frames
    """

    N = video_info['length'][()] # scalar, video original length
    cps = video_info['change_points'][()] # shape [n_segments,2], stores begin and end of each segment in original length index

    pred_score = pred_score.to("cpu").detach().numpy() # GPU->CPU, requires_grad=False, to numpy
    
    if inter_method == "upsample":
        pred_score = upsample(pred_score, N)
    elif inter_method == "cut":
        pred_score = cut(pred_score, N)

    pred_score_key_frames = (pred_score > 0.5) # convert to key frames

    value = np.array([pred_score_key_frames[cp[0]:(cp[1])].mean() for cp in cps]) # [n_segments]
    # weight = video_info['n_frame_per_seg'][()] # shape [n_segments], number of frames in each segment
    weight = np.ones((cps.shape[0]), dtype=int)

    _, selected = knapsack(list(zip(value, weight)), 2) 
    # _, selected = knapsack(list(zip(value, weight)), int(0.15*N)) # selected -> [66, 64, 51, 50, 44, 41, 40, 38, 34, 33, 31, 25, 24, 23, 20, 10, 9]
    selected = selected[::-1] # inverse the selected list, which seg is selected
    key_shots = np.zeros(shape=(N, ))
    for i in selected:
        key_shots[cps[i][0]:(cps[i][1])] = 1 # assign 1 to seg
        
    return pred_score.tolist(), key_shots

def upsample(pred_score, N):
    """
    Use Nearest Neighbor to extend from 320 to N
    input: 
        pred_score: shape [320], indicates key frame prob.
        N: scalar, video original length
    output
        up_arr: shape [N]
    """
    x = np.linspace(0, len(pred_score)-1, len(pred_score))
    f = interpolate.interp1d(x, pred_score, kind='nearest')
    x_new = np.linspace(0, len(pred_score)-1, N); #print(x_new, N)
    up_arr = f(x_new)

    return up_arr

def cut(arr, N):
    return arr[:N]

def rankcorrelation_kendall(y_pred, y_true):
    return kendalltau(rankdata(-y_true), rankdata(-y_pred))[0]

def rankcorrelation_spearman(y_pred, y_true):
    return spearmanr(y_true, y_pred)


if __name__ == "__main__":
    device = torch.device("cuda:0")

    import h5py
    data_file = h5py.File("datasets/fcsn_tvsum.h5")
    video_info = data_file["video_1"]
    pred_score = torch.randn((320,), requires_grad=True)
    pred_score = pred_score.to(device)
    select_keyshots(video_info, pred_score)
