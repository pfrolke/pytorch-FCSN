import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import optimizer
from tqdm.std import tqdm

from data_loader import get_loader
from FCSN import *
import eval_tools
import sys

# Hyperparameters
BATCH_SIZE = 3
LR = 1e-3
MOMENTUM = 0.9
EPOCHS = 100
INTER_METHOD = "cut"

# load training and testing dataset
train_loader_list, test_dataset_list, data_file = get_loader(
    "../fcsn_breakfast_summarization_dataset_keyframe.hdf5", "1D", BATCH_SIZE, 5)
# device use for training and testing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# array for calc. eval fscore
fscore_arr = np.zeros(len(train_loader_list))

model_num = len(os.listdir("scores"))
data_name = "breakfast"


def batch_loss(outputs, labels, class_weights):
    classification_loss_sum = 0

    cur_batch_size = labels.shape[0]
    for k in range(cur_batch_size):
        # get one output / label pair
        cur_outputs = outputs[k, :, :].permute(1, 0)  # [2,320]->[320,2]
        cur_label = labels[k, :]  # [320]

        criterion = nn.NLLLoss(weight=class_weights)

        log_p = torch.log(cur_outputs)

        classification_loss_sum += criterion(log_p, cur_label)

    return classification_loss_sum / cur_batch_size


def compute_class_weights(train_set):
    class_sample_count = np.zeros(2)
    n_samples = 0
    for _, label, _ in train_set:
        class_sample_count += np.unique(label, return_counts=True)[1]
        n_samples += label.shape[0]

    # "freq_c is the number of frames with label c divided by the total number of frames in videos where label c is present"
    freq_c = class_sample_count / n_samples
    median_freq = np.median(freq_c) 
    class_weights = median_freq / freq_c

    return torch.tensor(class_weights, device=device, dtype=torch.float)


for i in trange(len(train_loader_list), desc="split", leave=False):
    # plotting
    plt_epochs = []
    plt_losses = []
    plt_t_losses = []
    plt_t_epochs = []

    # model declaration
    model = FCSN_1D_sup()
    # optimizer declaration
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # optimizer = optim.Adam(model.parameters())

    # put model in to device
    model.to(device)

    class_weights = compute_class_weights(train_loader_list[i])

    for epoch in trange(EPOCHS, desc="epoch", leave=False):
        model.train()
        train_losses = []
        for batch_i, (feature, label, _) in enumerate(train_loader_list[i]):
            feature = feature.to(device)  # [5,1024,320]
            label = label.to(device)  # [5,320]
            outputs = model(feature)  # output shape [5,2,320]

            # zero the parameter gradients
            optimizer.zero_grad()

            total_loss = batch_loss(outputs, label, class_weights)

            train_losses.append(total_loss.item())

            total_loss.backward()
            optimizer.step()

        # plotting loss
        train_loss_avg = np.mean(train_losses)
        plt_epochs.append(epoch)
        plt_losses.append(train_loss_avg)

        # eval every 5 epoch
        if(epoch+1) % 5 == 0:
            eval_res_avg = []  # for all testing video results
            model.eval()

            score_dict = {}

            for j, (feature, label, index) in enumerate(test_dataset_list[i], 1):
                # [1024,320] -> [1,1024,320]
                feature = feature.view(1, 1024, -1).to(device)

                # [1,2,320] -> [320]
                pred_score = model(feature).view(-1, label.shape[0])[1]
                
                video_name = list(data_file.keys())[index]
                video_info = data_file[video_name]

                # select key shots by video_info and pred_score
                # pred_summary: [N] binary keyshot
                pred_score_upsample, pred_summary = eval_tools.select_keyshots(
                    video_info, pred_score, inter_method=INTER_METHOD)

                score_dict[video_name] = pred_score_upsample

                # [n_users,N]
                true_summary_arr = video_info['user_summary'][()]
                eval_res = [eval_tools.eval_metrics(
                    pred_summary, true_summary) for true_summary in true_summary_arr]  # [n_user,3]

                # mean of [precision, recall, fscore]
                eval_res = np.mean(eval_res, axis=0).tolist()

                # kendall = eval_tools.rankcorrelation_kendall(
                #     np.array(pred_score_upsample), video_info['gtscore'][()])
                kendall = 0
                eval_res_avg.append(np.append(eval_res, kendall))

            eval_res_avg = np.mean(eval_res_avg, axis=0)

            tqdm.write(
                f"split:{i} epoch:{epoch:0>3d} precision:{eval_res_avg[0]:.1%} recall:{eval_res_avg[1]:.1%} fscore:{eval_res_avg[2]:.1%} kendalltau: {eval_res_avg[3]:.3f} loss:{train_loss_avg:.4f}")

        # store the last fscore for eval, and remove model from GPU
        if((epoch+1) == EPOCHS):
            # store model
            torch.save(model.state_dict(
            ), f"trained/{model_num}-{data_name}-split_{i}-epoch_{epoch}.pkl")

            # store scores
            json_data = score_dict
            if os.path.exists(f"scores/{model_num}-{data_name}.json"):
                with open(f"scores/{model_num}-{data_name}.json", 'r') as f:
                    json_data.update(json.load(f))

            with open(f"scores/{model_num}-{data_name}.json", 'w') as f:
                json.dump(json_data, f)

            # store fscore
            fscore_arr[i] = eval_res_avg[2]

            # release model from GPU
            model = model.cpu()
            torch.cuda.empty_cache()

    plt.title("Epoch loss")
    plt.xlabel("Epochs")
    plt.ylabel("Mean loss")
    plt.ylim(0, 1)
    plt.plot(plt_epochs, plt_losses)
    plt.savefig(f"losses/{model_num}-{data_name}-split_{i}.pdf")
    plt.clf()


# print eval fscore
print("{} average fscore:{:.1%}".format(data_name, fscore_arr.mean()))
