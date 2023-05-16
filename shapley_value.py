""" save the region id and sampled orders
also calculate the Shapley value for each region (at the original position of the point cloud) """
import os
import importlib
import argparse
import torch
from data_shapley import ModelNet_Loader_Shapley_test
import numpy as np
from torch.utils.data import DataLoader
from tools.final_util import IOStream,set_random, get_folder_name_list, square_distance, mkdir
from tools.final_util import NUM_POINTS, NUM_REGIONS, NUM_SAMPLES_SAVE  # constants
from tools.final_common import cal_reward
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists(args.exp_folder):
        os.makedirs(args.exp_folder)

def cal_region_id(data, fps_index, result_path, save=True):
    """ calculate and save region id of all the points
    Input:
        data: (B,num_points,3) tensor, point cloud, num_points=1024
        fps_index: (num_regions,) ndarray, center idx of the 32 regions
        result_path: path to save file
    Return:
        region_id: (num_points,) ndarray, record each point belongs to which region
    """
    data_fps = data[:, fps_index, :] 
    distance = square_distance(data, data_fps) 
    region_id = torch.argmin(distance, dim=2) 
    region_id = region_id.squeeze().cpu().numpy() 
    if save:
        np.save(result_path + "region_id.npy", region_id)
    return region_id



def cal_norm_factor(model, data, lbl, center, result_path, args, save=True):
    """ calculate v(N) - v(empty)
    Input:
        data: (B,num_points,3) tensor, point cloud (already transposed)
        lbl: (B,) tensor, label
        center: (3,) tensor, center of point cloud
        result_path: path to save
    Return:
        norm_factor: scalar, v(Omega) - v(empty)
    """
    B = data.shape[0]
    empty = center.view(1, 1, 3).expand(B, args.num_points, 3).clone()
    v_N, _ = cal_reward(model, data, lbl, args)
    v_empty, _ = cal_reward(model, empty, lbl, args)
    norm_factor = (v_N - v_empty).cpu().item()
    if save:
        np.save(result_path + "norm_factor.npy", norm_factor)
    return norm_factor


def generate_all_orders(result_path, args, save=True):
    """ generate random orders for sampling
    Input:
        result_path: path to save all orders
    Return:
        all_orders: (num_samples_save, num_regions) ndarray
    """
    all_orders = []
    for k in range(args.num_samples_save):
        all_orders.append(np.random.permutation(np.arange(0, args.num_regions, 1)).reshape((1, -1)))
    all_orders = np.concatenate(all_orders, axis=0) 
    if save:
        np.save(result_path + "all_orders.npy", all_orders)
    return all_orders

def mask_data(masked_data, center, order, region_id):
    """ mask the data to the center of the point cloud
    Input:
        masked_data: (region+1, num_points,3) tensor, data to be masked
        center: (3,) tensor, center of point cloud
        order: (num_regions,) ndarray
        region_id: (num_points,) ndarray
    Return:
        masked_data: (region+1, num_points,3) tensor, modified
    """

    for j in range(1, len(order) + 1):
        mask_region_id = order[j - 1]  
        mask_index = (region_id == mask_region_id)  
        masked_data[:j, mask_index, :] = center     
    return masked_data


def save_shapley(region_shap_value, pc_idx, count, result_path, region_id, args):
    N = args.num_points
    shap_value = np.zeros((N,))
    folder = result_path + "shapley/"
    mkdir(folder)
    folder2 = result_path + "region_shapley/"
    mkdir(folder2)

    for k in range(0, args.num_regions):
        region_index = (region_id == k) 
        shap_value[region_index] = region_shap_value[k] / count  

    np.save(folder + "%s.npy" % (str(pc_idx) + '_' + str(count)), shap_value) 
    np.save(folder2 + "%s.npy" % (str(pc_idx) + '_' + str(count)), region_shap_value / count)


def shap_sampling(model, dataloader, args, folder_name_list):
    sample_nums = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
    with torch.no_grad():
        model.eval()
        fps_indices = np.load('fps_%s_%d_%d_center_index.npy'%(args.dataset, args.num_points, args.num_regions))
        fps_indices = torch.from_numpy(fps_indices).to(args.device)

        for i, (data, lbl, normal) in enumerate(dataloader): 
            B, N = data.shape[0], args.num_points
            folder_name = folder_name_list[i]
            result_path = args.exp_folder + '%s/' % folder_name
            mkdir(result_path)
            count = 0
            region_sv_all = []  
            region_shap_value = np.zeros((args.num_regions,)) 
            data = data.to(args.device) 
            lbl = lbl.to(args.device) 
            fps_index = fps_indices[i] 
            region_id = cal_region_id(data, fps_index, result_path, save=True) 
            center = torch.mean(data, dim=1).squeeze() 
            norm_factor = cal_norm_factor(model, data, lbl, center, result_path, args, save=True)
            all_orders = generate_all_orders(result_path, args, save=True) 

            while count < args.num_samples_save:
                order = all_orders[count]  
                masked_data = data.expand(args.num_regions + 1, N, 3).clone() 
                masked_data = mask_data(masked_data, center, order, region_id)
                v, _ = cal_reward(model, masked_data, lbl, args)
                dv = v[1:] - v[:-1]  
                region_shap_value[order] += (dv.cpu().numpy()) 
                temp = np.zeros((args.num_regions,))
                temp[order] += dv.cpu().numpy() 
                region_sv_all.append(temp) 
                count += 1

                if count == 1000:
                    print("pointcloud:%s, index:%d, sample:%d" % (folder_name, i, count))
                #if count in sample_nums:
                    save_shapley(region_shap_value, i, count, result_path, region_id, args)

            np.save(result_path + "region_sv_all.npy", region_sv_all) # (num_samples_save, num_regions)


def test(args):

    if args.dataset == "modelnet40":
        data_loader = DataLoader(ModelNet_Loader_Shapley_test(args, partition='train', num_points=args.num_points),
                                 num_workers=1,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == "shapenet":
        print('no shapenet')
        '''
        data_loader = DataLoader(ShapeNetDataset_Shapley_test(args, split='train', npoints=args.num_points,
                                                              class_choice=SHAPENET_CLASS, classification=True),
                                 num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)'''
    else:
        raise Exception("Dataset does not exist")


    '''MODEL LOADING '''  
    experiment_dir = 'log/classification/pointnet_cls_0306' 
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    folder_name_list = get_folder_name_list(args) 
    shap_sampling(classifier, data_loader, args, folder_name_list)


def main():
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--model', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'pointnet2'])
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40', 'shapenet'])
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--device_id', type=int, default=1, help='gpu id to use')  # change GPU here
    parser.add_argument('--softmax_type', type=str, default="modified", choices=["normal", "modified"])
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')

    args = parser.parse_args()

    args.num_points = NUM_POINTS
    args.num_regions = NUM_REGIONS
    args.num_samples_save = NUM_SAMPLES_SAVE  #1000
    args.exp_folder = './checkpoints/exp_%s_%s_%d_%d_shapley/' % (
        args.model, args.dataset, args.num_points, args.num_regions) 
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    _init_(args)
    set_random(args.seed)

    if args.cuda:
        print(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    else:
        print('Using CPU')

    test(args)


if __name__ == "__main__":
    main()
