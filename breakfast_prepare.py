import h5py
import numpy as np
from shutil import copy2
import os
from tqdm import tqdm
from sklearn.metrics import f1_score

def downsample(arr, N):
    old_N = len(arr)
    return np.array([arr[int((i/N)*old_N)] for i in range(N)])

def generate_gt(vid_data):
    user_summary = vid_data['user_summary'][()]
    # user_summary = np.array([downsample(usr, 48) for usr in user_summary])

    _, vid_len = user_summary.shape

    # sort the frames by priority
    priority_idx = np.argsort(-(user_summary.sum(axis=0)))

    gt_sum = np.zeros((vid_len))
    max_fscore = -1
    for f in tqdm(priority_idx, desc="generating ground truth", leave=False):
        gt_sum[f] = 1

        cur_fscore = np.mean([f1_score(usr, gt_sum) for usr in user_summary])

        if cur_fscore > max_fscore:
            max_fscore = cur_fscore
        else:
            gt_sum[f] = 0

    tqdm.write(f"f1-score: {max_fscore}")

    # old_len = vid_data['n_frames'][()]
    # new_len = 48.0
    # ratio = new_len / old_len

    # convert to key-frames by taking middle frame of each segment
    mid_frames = np.mean(vid_data['change_points'][()], axis=1, dtype=int)
  
    gt_sum_kf = np.array(
        [1 if f in mid_frames and gt_sum[f] else 0 for f in range(vid_len)])

    return gt_sum, gt_sum_kf

BF_PATH = "breakfast_summarization_dataset.hdf5"
FCSN_BF_PATH = "fcsn_breakfast_summarization_dataset_keyframe.hdf5"

if os.path.exists(FCSN_BF_PATH):
    os.remove(FCSN_BF_PATH)
copy2(BF_PATH,
      FCSN_BF_PATH)

with h5py.File(FCSN_BF_PATH, "a") as bf_data:
    for item in tqdm(bf_data):
        tqdm.write(item)
        vid_data = bf_data[item]

        # fix change point bug for P03_webcam02_P03_friedegg
        if item == "P03_webcam02_P03_friedegg":
            cps = vid_data["change_points"][()]
            nfps = vid_data["n_frame_per_seg"][()]

            del vid_data["change_points"]
            del vid_data["n_frame_per_seg"]

            vid_data["change_points"] = cps[:-1]
            vid_data["n_frame_per_seg"] = nfps[:-1]

        # add length
        vid_len = vid_data['n_frames'][()]
        vid_data['length'] = vid_len

        # add features
        old_features = vid_data['features'][()]
        # feature = np.array([old_features[int((i/48) * vid_len)] for i in range(48)], dtype=np.float32)

        # feature[:vid_len] = vid_data['features'][()]
        vid_data['feature'] = old_features

        # add ground truth labels
        gt_sum, gt_sum_kf = generate_gt(vid_data)
        # label = np.zeros((288))
        # label[:vid_len] = gt_sum
        vid_data['label'] = gt_sum_kf

        del vid_data['gtsummary']
        vid_data['gtsummary'] = gt_sum
