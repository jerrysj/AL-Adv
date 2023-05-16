import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pytorch3d.ops import knn_points, knn_gather

NUM_POINTS = 1024  # number of points in each point cloud
NUM_REGIONS = 32  # number of regions in each point cloud
NUM_SAMPLES_SAVE = 1000  # number of random permutations to save at initial state

DATA_MODELNET_SHAPLEY_TEST = 'modelnet40_test_adv.txt'



def set_random(seed):
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)   
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)   
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False  


def get_folder_name_list(args):
    """ get names of all samples for Shapley value test
    """
    if args.dataset == "modelnet40":
        f = open(os.path.join('misc', DATA_MODELNET_SHAPLEY_TEST), 'r')
        names = [str.rstrip() for str in f.readlines()]
        f.close()
    elif args.dataset == "shapenet":
        print('no shapenet')

    else:
        raise Exception("Dataset does not exist")

    return names

def square_distance(src, dst):
    """ Calculate Euclid distance between each two points.
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


# attack
def normalize(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

def get_kappa_ori(pc, normal, k=2):
    b,_,n=pc.size()
    #inter_dis = ((pc.unsqueeze(3) - pc.unsqueeze(2))**2).sum(1)
    #inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()
    #nn_pts = torch.gather(pc, 2, inter_idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)
    inter_KNN = knn_points(pc.permute(0,2,1), pc.permute(0,2,1), K=k+1) #[dists:[b,n,k+1], idx:[b,n,k+1]]
    nn_pts = knn_gather(pc.permute(0,2,1), inter_KNN.idx).permute(0,3,1,2)[:,:,:,1:].contiguous() # [b, 3, n ,k]
    vectors = nn_pts - pc.unsqueeze(3)
    vectors = normalize(vectors)

    return torch.abs((vectors*normal.unsqueeze(3)).sum(1)).mean(2) # [b, n]

def get_kappa_adv(adv_pc, ori_pc, ori_normal, k=2):
    b,_,n=adv_pc.size()
    # compute knn between advPC and oriPC to get normal n_p
    #intra_dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    #intra_idx = torch.topk(intra_dis, 1, dim=2, largest=False, sorted=True)[1]
    #normal = torch.gather(ori_normal, 2, intra_idx.view(b,1,n).expand(b,3,n))
    intra_KNN = knn_points(adv_pc.permute(0,2,1), ori_pc.permute(0,2,1), K=1) #[dists:[b,n,1], idx:[b,n,1]]
    normal = knn_gather(ori_normal.permute(0,2,1), intra_KNN.idx).permute(0,3,1,2).squeeze(3).contiguous() # [b, 3, n]

    # compute knn between advPC and itself to get \|q-p\|_2
    #inter_dis = ((adv_pc.unsqueeze(3) - adv_pc.unsqueeze(2))**2).sum(1)
    #inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()
    #nn_pts = torch.gather(adv_pc, 2, inter_idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)
    inter_KNN = knn_points(adv_pc.permute(0,2,1), adv_pc.permute(0,2,1), K=k+1) #[dists:[b,n,k+1], idx:[b,n,k+1]]
    nn_pts = knn_gather(adv_pc.permute(0,2,1), inter_KNN.idx).permute(0,3,1,2)[:,:,:,1:].contiguous() # [b, 3, n ,k]
    vectors = nn_pts - adv_pc.unsqueeze(3)
    vectors = normalize(vectors)

    return torch.abs((vectors*normal.unsqueeze(3)).sum(1)).mean(2), normal # [b, n], [b, 3, n]

def curvature_loss(adv_pc, ori_pc, adv_kappa, ori_kappa, k=2):
    b,_,n=adv_pc.size()

    # intra_dis = ((input_curr_iter.unsqueeze(3) - pc_ori.unsqueeze(2))**2).sum(1)
    # intra_idx = torch.topk(intra_dis, 1, dim=2, largest=False, sorted=True)[1]
    # knn_theta_normal = torch.gather(theta_normal, 1, intra_idx.view(b,n).expand(b,n))
    # curv_loss = ((curv_loss - knn_theta_normal)**2).mean(-1)

    intra_KNN = knn_points(adv_pc.permute(0,2,1), ori_pc.permute(0,2,1), K=1) #[dists:[b,n,1], idx:[b,n,1]]
    onenn_ori_kappa = torch.gather(ori_kappa, 1, intra_KNN.idx.squeeze(-1)).contiguous() # [b, n]

    curv_loss = ((adv_kappa - onenn_ori_kappa)**2).mean(-1)

    return curv_loss

def compare(output, target, gt, targeted):
    if targeted:
        return output == target
    else:
        return output != gt

def chamfer_loss(adv_pc, ori_pc):
    # Chamfer distance (two sides)
    #intra_dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    #dis_loss = intra_dis.min(2)[0].mean(1) + intra_dis.min(1)[0].mean(1)
    adv_KNN = knn_points(adv_pc.permute(0,2,1), ori_pc.permute(0,2,1), K=1) #[dists:[b,n,1], idx:[b,n,1]]
    ori_KNN = knn_points(ori_pc.permute(0,2,1), adv_pc.permute(0,2,1), K=1) #[dists:[b,n,1], idx:[b,n,1]]
    dis_loss = adv_KNN.dists.contiguous().squeeze(-1).mean(-1) + ori_KNN.dists.contiguous().squeeze(-1).mean(-1) #[b]
    return dis_loss

def hausdorff_loss(adv_pc, ori_pc):
    #dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    #hd_loss = torch.max(torch.min(dis, dim=2)[0], dim=1)[0]
    adv_KNN = knn_points(adv_pc.permute(0,2,1), ori_pc.permute(0,2,1), K=1) #[dists:[b,n,1], idx:[b,n,1]]
    hd_loss = adv_KNN.dists.contiguous().squeeze(-1).max(-1)[0] #[b]
    return hd_loss