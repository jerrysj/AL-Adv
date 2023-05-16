import torch
import os
import numpy as np
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from tools.final_util import set_random, get_folder_name_list
from tools.final_util import NUM_POINTS, NUM_REGIONS
from data_shapley import ModelNet_Loader_Shapley_test
import importlib
import sys
from tqdm import tqdm
from torch.autograd import Variable
import random

cls_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone',
             'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
             'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
             'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']


def inference(classifier, points):
    points = points.transpose(2, 1)  
    p, _ , _= classifier(points)  
    p_choice = p.data.max(-1)[1]  
    p[torch.arange(points.shape[0]), p_choice] = -1000.0
    s_idx = p.data.max(-1)[1]
    return p_choice, s_idx


def inference_2nd(pred, points, target):
    pred[torch.arange(points.shape[0]), target.long()] = -1000.0
    s_idx = pred.data.max(-1)[1]
    return s_idx


def square_distance(src, dst): 
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

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


def chamfer_loss_me(adv_points, points):
    adv_dist = square_distance(adv_points, points)
    adv_min_dist = adv_dist.min(-1)[0]  
    adv_ch_loss = adv_min_dist.mean(-1).mean(-1)

    ori_dist = square_distance(points, adv_points)
    ori_min_dist = ori_dist.min(-1)[0]
    ori_ch_loss = ori_min_dist.mean(-1).mean(-1)

    ch_loss = torch.max(adv_ch_loss, ori_ch_loss).item()
    return ch_loss


def hausdorff_loss_me(adv_points, points):
    ori_dist = square_distance(points, adv_points) 
    ori_min_dist = ori_dist.min(-1)[0]  
    ori_ha_loss = ori_min_dist.max(-1)[0].mean(-1)  

    adv_dist = square_distance(adv_points, points)
    adv_min_dist = adv_dist.min(-1)[0]
    adv_ha_loss = adv_min_dist.max(-1)[0].mean(-1)

    ha_loss = torch.max(ori_ha_loss, adv_ha_loss).item()
    return ha_loss


def count_grad(classifier, criterion, points, target):
    p = points.detach().clone().cuda().requires_grad_(True)  
    p_t = p.transpose(2, 1)  
    pred, trans_feat, _ = classifier(p_t) 
    loss = criterion(pred, target.long(), trans_feat) 
    loss.backward(retain_graph=True)
    classifier.zero_grad()  
    grad = p.grad.detach()  
    return grad, loss


def adv_propagation_cw(classifier, criterion, points, target, args, region_id, region_id_list): 
    step_num = 500
    lmd = 0.2
    is_suc = False
    B = points.shape[0]
    N = points.shape[1]
    pre_label = None
    adv_best = None
    ar = None

    adv_samples = torch.zeros(points.shape).cuda().requires_grad_(True) 
    optimizer = torch.optim.Adam(
        [adv_samples],
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
    )

    idx = torch.zeros(B, N, dtype=torch.long)
    for num2 in range(len(region_id_list)):
        region_index = (region_id_list[num2] == region_id) 
        idx[:, region_index] = 1

    idx = idx.cuda()
    num = torch.sum(idx, -1) 
    total_num = points.shape[1]
    selected = num.float() / float(total_num)
    np.set_printoptions(precision=2)  
    print("selected points ratio: {}".format(selected.cpu().numpy()))

    adv_points = points.clone() 

    lower_bound = torch.ones(1).cuda() * 0  
    scale_const = torch.ones(1).cuda() * 50  
    upper_bound = torch.ones(1).cuda() * 1e10  
   
    best_loss = []
    adv_points_best = []
    adv_count = []
    for j in range(10):
        iter_suc = False
        iter_best_loss = 1e10 
        iter_dist_loss = 1e10
        adv_points_iter = None
        iter_best_count = None
        for i in range(step_num):
            pp, loss_net = count_grad(classifier, criterion, adv_points, target)
            samples = adv_samples.mul(idx.float().unsqueeze(-1).expand(points.shape))  # {Tensor:(1,1024,3)}
            rationx = torch.abs(pp[:, :, 0]) / (torch.abs(pp[:, :, 0]) + torch.abs(pp[:, :, 1]) + torch.abs(pp[:, :, 2]) + 1e-10)
            rationy = torch.abs(pp[:, :, 1]) / (torch.abs(pp[:, :, 0]) + torch.abs(pp[:, :, 1]) + torch.abs(pp[:, :, 2]) + 1e-10)
            rationz = torch.abs(pp[:, :, 2]) / (torch.abs(pp[:, :, 0]) + torch.abs(pp[:, :, 1]) + torch.abs(pp[:, :, 2]) + 1e-10)
            ration = torch.cat([rationx, rationy, rationz], dim=0).t().unsqueeze(0)
            sign_grad = 0.6 * torch.sign(pp) * ration
            adv_points = points + sign_grad * samples  # {Tensor:(1,1024,3)}
          

            adv_points.requires_grad_(True)
            pred, trans_feat, _ = classifier(adv_points.transpose(2, 1))

            cls_pred = pred[torch.arange(B), target.long()]  
            adv_target = inference_2nd(pred.clone().detach(), adv_points.transpose(2, 1), target)
            adv_cls_pred = pred[torch.arange(B), adv_target.long()]  
            # h(P'):max(0, cls_pred-adv_cls_pred)
            h_p = torch.max(torch.tensor(0.0).cuda(),
                            (torch.mean(torch.exp(cls_pred)) - torch.mean(torch.exp(adv_cls_pred))).cuda())

            l_dist = torch.sum(samples[..., :3] ** 2, -1).mean(-1).mean(-1) 
            l2_loss = torch.max(l_dist, torch.tensor(lmd * 0.002).cuda()) 

            # hausdorff_loss
            hs_dist = hausdorff_loss_me(adv_points, points)
            hs_dist_loss = torch.max(torch.tensor(hs_dist).cuda(), torch.tensor(lmd * 0.002).cuda())

            # chamfer_loss
            ch_dist = chamfer_loss_me(adv_points, points)  # points:{Tensor:(1,1024,3)}
            ch_dist_loss = torch.max(torch.tensor(ch_dist).cuda(), torch.tensor(lmd * 0.002).cuda())

            a1 = points.eq(adv_points)  
            a2 = a1.view(N, 3)
            a3 = torch.sum(a2, -1)
            arfa = (a3 < 3)
            ar = torch.sum(arfa)  
            
            dist_loss = ch_dist_loss + hs_dist_loss
            loss = h_p + scale_const * dist_loss + 0.3 * ar  # {Tensor:()}

            if i % 100 == 0:
                print(
                    "EPOCH:{}, L2_LOSS:{:.10f}, HS_LOSS:{:.10f}, CF_LOSS:{:.10f}, hp_LOSS:{:.6f}, perturb_count:{:.1f}, TOTAL_LOSS:{:.6f}"
                        .format(i,
                                l_dist.detach().cpu().item(),
                                hs_dist,
                                ch_dist,
                                h_p.detach().cpu().item(),
                                ar.item(),
                                loss.detach().cpu().item()))

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # is attack success
            pre_label = torch.argmax(pred).item()  
            if pre_label != target.item():  
                is_suc = True
                iter_suc = True
                if loss < iter_best_loss:
                    iter_best_loss = loss
                    iter_best_count = ar.item()
                    iter_dist_loss = dist_loss
                    adv_points_iter = adv_points.clone()

        best_loss.append(iter_dist_loss)
        adv_points_best.append(adv_points_iter)
        adv_count.append(iter_best_count)

        if iter_suc:
            lower_bound = max(lower_bound, scale_const)
            if upper_bound < 1e9:
                scale_const = (lower_bound + upper_bound) * 0.5
            else:
                scale_const *= 2
        else:
            upper_bound = min(upper_bound, scale_const)
            if upper_bound < 1e9:
                scale_const = (lower_bound + upper_bound) * 0.5

    if is_suc:
        best_loss = torch.tensor(best_loss, device='cpu')
        best_index = np.argmin(best_loss)
        return is_suc, adv_points_best[best_index], pre_label, adv_count[best_index]
    else:
        return is_suc, adv_points, pre_label, ar.item()


def test_attack(model, loader, criterion, args):
    CH_list = []
    HS_list = []
    num_sample = 0
    num_attack_success = 0
    perturb_count_ar = 0
    num_orierror = 0
    for j, (data, lbl, normal) in tqdm(enumerate(loader), total=len(loader)):  # data:{Tensor:(1,1024,3)},lbl:{Tensor:(1,)}
        pcdet = data.permute(0, 2, 1).cuda()
        det_output, _ ,_= model(pcdet)  # {Tensor:(1,40)}
        output_label = torch.argmax(det_output).item()  # output_label:{int}
        if output_label != lbl.item():  
            num_orierror += 1
            continue

        num_sample += 1
        name = folder_name_list[j]
        print('attack to: %s' % name)
        base_folder = args.exp_folder + '%s/' % name

        saved_adv_path = os.path.join('./PR_result', 'AL-Adv-count')
        if not os.path.exists(saved_adv_path):
            os.makedirs(saved_adv_path)

        # regoin_id
        region_id = np.load(base_folder + "region_id.npy") #{ndarray:(1024,)}

        # perturb region
        region_shapley = np.load(base_folder + "region_shapley/%d_1000.npy" % (int(j)))  #{ndarray:(32,)}
        region_index = np.argsort(region_shapley)  
        region_id_list = []
        for num in range(args.num_perturb):
            idex = num + 1
            region_id_list.append(region_index[-idex])

        print("BATCH: %d" % j)
        points = data  # {Tensor:(1,1024,3)}
        target = lbl  # {Tensor:(1,)}:tensor([0])
        points, target = points.cuda(), target.cuda()
        is_suc, adv_points, pred_class, ar_count = adv_propagation_cw(model.eval(), criterion, points, target, args,
                                                                   region_id, region_id_list) # adv_points:{Tensor:(1,1024,3)}

        pcdet = adv_points.permute(0, 2, 1)
        det_output, _ , _  = model(pcdet)  # {Tensor:(1,40)}
        output_label = torch.argmax(det_output).item()  # output_label:{int}
        
        if output_label != lbl.item():  # attack success
            num_attack_success += 1
            print('Success: %s' % name)
            perturb_count_ar += ar_count
            CH_dist = chamfer_loss_me(adv_points, points)  # points:{Tensor:(1,1024,3)}
            HS_dist = hausdorff_loss_me(adv_points, points)
            CH_list.append(CH_dist)
            HS_list.append(HS_dist)
        else:
            print('False: %s' % name)

    ch_dist_mean = 1.0 * np.sum(CH_list) / (len(CH_list))
    hs_dist_mean = 1.0 * np.sum(HS_list) / (len(HS_list))
    success_rate = num_attack_success / num_sample
    count_ar = perturb_count_ar / num_attack_success

    return success_rate, ch_dist_mean, hs_dist_mean, count_ar


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = BASE_DIR
    sys.path.append(os.path.join(ROOT_DIR, 'models'))

    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--model', type=str, default='pointnet', metavar='N')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N', choices=['modelnet40', 'shapenet'])
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--device_id', type=int, default=1)
    parser.add_argument('--num_category', default=40, type=int, choices=[40], help='training on ModelNet40')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--num_perturb', type=int, default=1, help='The number of disturbed regions')

    args = parser.parse_args()

    args.num_points = NUM_POINTS
    args.num_regions = NUM_REGIONS
    args.exp_folder = './checkpoints/exp_%s_%s_%d_%d_shapley/' % (
        args.model, args.dataset, args.num_points, args.num_regions) 
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    set_random(args.seed)
    folder_name_list = get_folder_name_list(args) 

    if args.dataset == "modelnet40":
        args.data_loader = DataLoader(ModelNet_Loader_Shapley_test(args, partition='train', num_points=args.num_points),
                                      num_workers=1,
                                      batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    elif args.dataset == "shapenet":
        print('no shapenet')
    else:
        raise Exception("Dataset does not exist")

    experiment_dir = 'log/classification/pointnet_cls_0306'  
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)
    floss = model.get_loss().cuda()
    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    success_rate, ch_dist_mean, hs_dist_mean, count = test_attack(classifier, args.data_loader, floss, args)

    print('###############result#################')
    print('Success Rate: %f ' % success_rate)
    print('Chamfer distance: ', "%e" % ch_dist_mean)
    print('Hausdorff distance: ', "%e" % hs_dist_mean)
    print('#count: %f ' % count)
