import os
import math
from tqdm import tqdm
import numpy as np
import random
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.config import Config
import numbers
import math
import cv2
import h5py



def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))



def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model





def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=0, keepdims=True)
    s = x_exp / (x_sum)
    return s

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

def maybe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_nifti(path):
    itk_img = sitk.ReadImage(path)

    itk_arr = sitk.GetArrayFromImage(itk_img)
    return itk_arr



def read_list(split, task):

    config = Config(task)
    ids_list = np.loadtxt(
        os.path.join(config.save_dir, 'splits_txts', f'{split}.txt'),
        dtype=str
    ).tolist()
    return sorted(ids_list)


def read_data(data_id, task, normalize=False):
    config = Config(task)
    im_path = os.path.join(config.save_dir, 'npy', f'{data_id}_image.npy')
    lb_path = os.path.join(config.save_dir, 'npy', f'{data_id}_label.npy')
    if not os.path.exists(im_path) or not os.path.exists(lb_path):
        print(im_path)
        print(lb_path)
        raise ValueError(data_id)
    image = np.load(im_path)
    label = np.load(lb_path)

    if normalize:
        if "synapse" in task:
            image = image.clip(min=-75, max=275)
        elif "mnms" in task:
            p5 = np.percentile(image.flatten(), 0.5)
            p95 = np.percentile(image.flatten(), 99.5)
            image = image.clip(min=p5, max=p95)

        image = (image - image.min()) / (image.max() - image.min())

        image = image.astype(np.float32)

    return image, label


def read_list_2d(split, task):

    config = Config(task+"_2d")
    ids_list = np.loadtxt(
        os.path.join(config.save_dir, 'split_txts', f'{split}.txt'),
        dtype=str
    ).tolist()
    return sorted(ids_list)


def read_data_2d(data_id, task, normalize=False):
    config = Config(task+"_2d")
    if "acdc" in task:
        h5File = h5py.File(os.path.join(config.save_dir, 'h5', f'{data_id}.h5'), 'r')
        image = h5File["image"][:]
        label = h5File["label"][:]
    else:
        im_path = os.path.join(config.save_dir, 'png', f'{data_id}_image.png')
        lb_path = os.path.join(config.save_dir, 'png', f'{data_id}_label.png')
        if not os.path.exists(im_path) or not os.path.exists(lb_path):
            print(im_path)
            print(lb_path)
            raise ValueError(data_id)
        image = cv2.imread(im_path, 0)
        label = cv2.imread(lb_path, 0)

    if normalize:
        if "synapse" in task:
            image = image.clip(min=-75, max=275)
        elif "mnms" in task:
            p5 = np.percentile(image.flatten(), 0.5)
            p95 = np.percentile(image.flatten(), 99.5)
            image = image.clip(min=p5, max=p95)

        image = (image - image.min()) / (image.max() - image.min())

        image = image.astype(np.float32)

    return image, label


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def fetch_data(batch, labeled=True):
    image = batch['image'].cuda()
    if labeled:
        label = batch['label'].cuda().unsqueeze(1)
        return image, label
    else:
        return image


def test_all_case(task, net, ids_list, num_classes, patch_size, stride_xy, stride_z, test_save_path=None):
    for data_id in tqdm(ids_list):
        image, _ = read_data(data_id, task=task, normalize=True)

        pred, _ = test_single_case(
            net,
            image,
            stride_xy,
            stride_z,
            patch_size,
            num_classes=num_classes
        )
        out = sitk.GetImageFromArray(pred.astype(np.float32))
        sitk.WriteImage(out, f'{test_save_path}/{data_id}.nii.gz')


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes):

    padding_flag = image.shape[0] < patch_size[0] or image.shape[1] < patch_size[1] or image.shape[2] < patch_size[2]
    if padding_flag:
        pw = max((patch_size[0] - image.shape[0]) // 2 + 1, 0)
        ph = max((patch_size[1] - image.shape[1]) // 2 + 1, 0)
        pd = max((patch_size[2] - image.shape[2]) // 2 + 1, 0)
        image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

    image = image[np.newaxis]
    _, dd, ww, hh = image.shape


    image = image.transpose(0, 3, 2, 1) #
    patch_size = (patch_size[2], patch_size[1], patch_size[0])
    _, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes, ) + image.shape[1:4]).astype(np.float32)
    cnt = np.zeros(image.shape[1:4]).astype(np.float32)
    for x in range(sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(sy):
            ys = min(stride_xy*y, hh-patch_size[1])
            for z in range(sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch = image[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                test_patch = test_patch.transpose(2, 4)
                y1, _ = net(test_patch)
                y = F.softmax(y1, dim=1) # <--
                y = y.cpu().data.numpy()
                y = y[0, ...]
                y = y.transpose(0, 3, 2, 1)
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1

    score_map = score_map / np.expand_dims(cnt, axis=0)
    score_map = score_map.transpose(0, 3, 2, 1)
    label_map = np.argmax(score_map, axis=0)
    return label_map, score_map
