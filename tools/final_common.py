import torch
import torch.nn as nn
import numpy as np
import time
from final_data_shapley import ModelNet_Loader_Shapley_test
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tools.final_util import IOStream, mkdir, get_folder_name_list


def get_reward(logits, lbl, args):
    """ given logits, calculate score for Shapley value or interaction
    Input:
        logits: (B', num_class) tensor, B' can be (num_region+1)*bs or B'=1
        lbl: (B,) tensor, B=1, label
    Return:
        v: (B',) tensor, reward score
    """
    num_class = logits.size()[1]
    if args.softmax_type == "normal":
        v = F.log_softmax(logits, dim=1)[:, lbl[0]]
    else: # args.softmax_type == "modified":
        v = logits[:, lbl[0]] - torch.logsumexp(logits[:, np.arange(num_class) != lbl[0].item()], dim=1)
    return v

def cal_reward(model, data, lbl, args):
    """ given data and label, calculate score for Shapley value or interaction
    Input:
        data: (B',num_points,3) tensor, B' can be (num_region+1)*bs or B=1, num_points=1024
        lbl: (B,) tensor, B=1, label
    Return:
        v: (B',) tensor, reward score
        logits: (B', num_class) tensor, logits for saving
    """


    data = data.permute(0,2,1).contiguous() #(B',3,num_points)
    if args.model == "pointnet" or args.model == "pointnet_roty_da":
        _, _, logits = model(data) # (B',num_class)
    else:
        logits = model(data)

    v = get_reward(logits, lbl, args)

    return v, logits


def mask_data_batch(masked_data, center, orders, region_id, args):
    """ mask the point cloud to center by region, implemented in batch
    Input:
        masked_data: ((num_regions + 1) * bs, num_points, 3) tensor, data to be masked
        center: (3,) tensor, center of the point cloud
        orders: (bs,num_regions) ndarray, a batch of orders for masking
        region_id: (num_points,) ndarray, record each point belongs to which region
    Return:
        masked_data: ((num_regions + 1) * bs, num_points, 3) tensor, modified
    """
    for o_idx, order in enumerate(orders): # for each order/permutation in the batch
        for j in range(1, len(order) + 1):
            mask_region_id = order[j - 1]
            mask_index = (region_id == mask_region_id)
            masked_data[(args.num_regions + 1) * o_idx:(args.num_regions + 1) * o_idx + j, mask_index, :] = center
    return masked_data


def shap_sampling_all_regions_batch(model, data_disturb, lbl, region_id, load_order_list, args):
    """ calculate shapley value for all regions on the disturbed point cloud
    Input:
        data_disturb: (B,num_points,3) tensor, B=1, num_points=1024, disturbed point cloud
        lbl: (B,) tensor, B=1
        region_id: (num_points,) ndarray, record that each point belongs to which region
        load_order_list: (num_samples_save, num_regions) ndarray, all orders saved previously
    Return:
        region_shap_value: (num_regions,) ndarray, shapley value of all regions
        all_logits_this_pose: (num_samples * (num_regions+1), num_class) tensor, saved logits
    """
    N = args.num_points
    num_regions = args.num_regions
    bs = args.shapley_batch_size
    iterations = args.num_samples // bs # number of batches we need to iterate through

    center = torch.mean(data_disturb, dim=1).squeeze()  # (3,) tensor

    with torch.no_grad():
        region_shap_value = np.zeros((num_regions,))
        all_logits_this_pose = [] # we also save the logits in case we need it for other use
        t_start = time.time()
        for i in range(iterations):
            orders = load_order_list[i * bs: (i + 1) * bs]
            masked_data = data_disturb.expand((num_regions + 1) * bs, N, 3).clone()
            masked_data = mask_data_batch(masked_data, center, orders, region_id, args)

            v, logits = cal_reward(model, masked_data, lbl, args) # v is ((num_regions + 1) * bs,), logits is ((num_regions + 1) * bs, num_class)
            all_logits_this_pose.append(logits)
            for o_idx, order in enumerate(orders):
                v_single_order = v[(num_regions + 1) * o_idx: (num_regions + 1) * (o_idx + 1)]  # (num_regions+1,)
                dv = v_single_order[1:] - v_single_order[:-1]
                region_shap_value[order] += (dv.cpu().numpy())
        region_shap_value /= args.num_samples
        all_logits_this_pose = torch.cat(all_logits_this_pose, dim=0) # (num_samples * (num_regions+1), num_class) tensor
        assert all_logits_this_pose.size()[0] == args.num_samples * (num_regions+1)

    t_end = time.time()
    print("done time: ", t_end-t_start)
    return region_shap_value, all_logits_this_pose
