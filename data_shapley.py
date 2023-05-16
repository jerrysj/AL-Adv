import os
import numpy as np
import torch
import json
from torch.utils.data import Dataset
from tools.final_util import DATA_MODELNET_SHAPLEY_TEST


def make_dataset_modelnet40(mode, opt):
    dataset = []

    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'modelnet40_normal_resampled')

    f = open(os.path.join(DATA_DIR, 'modelnet40_shape_names.txt')) 
    shape_list = [str.rstrip() for str in f.readlines()]
    f.close()

    if 'train' == mode:
        f = open(os.path.join("misc", DATA_MODELNET_SHAPLEY_TEST),'r') 
        lines = [str.rstrip() for str in f.readlines()]
        f.close()
    else:
        raise Exception('Network mode error.')

    for i, name in enumerate(lines):
        folder = name[0:-5]
        file_name = name
        label = shape_list.index(folder)
        item = (os.path.join(DATA_DIR, folder, file_name + '.txt'), label)
        dataset.append(item)

    return dataset




class ModelNet_Loader_Shapley_test(Dataset):
    def __init__(self, opt, num_points, partition='train'):
        super(ModelNet_Loader_Shapley_test, self).__init__()

        self.opt = opt
        self.partition = partition
        self.num_points = num_points

        self.dataset = make_dataset_modelnet40(self.partition, opt)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pc_np_file, class_id = self.dataset[index]
        data = np.loadtxt(pc_np_file, delimiter=',').astype(np.float32)
        pointcloud = data[0:self.num_points, 0:3]  # Nx3
        normal = data[0:self.num_points, 3:6]  # Nx3
        pointcloud = pointcloud.astype(np.float32)  # Nx3
        normal = normal.astype(np.float32)  # Nx3
        return pointcloud, class_id, normal



