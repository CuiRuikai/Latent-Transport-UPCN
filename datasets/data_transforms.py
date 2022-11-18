# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-02 14:38:36
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-07-03 09:23:07
# @Email:  cshzxie@gmail.com

import numpy as np
import torch
import transforms3d

class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
                'objects': tr['objects']
            })  # yapf: disable

    def __call__(self, data):
        for tr in self.transformers:
            transform = tr['callback']
            objects = tr['objects']
            rnd_value = np.random.uniform(0, 1)
            if transform.__class__ in [NormalizeObjectPose]:
                data = transform(data)
            else:
                for k, v in data.items():
                    if k in objects and k in data:
                        if transform.__class__ in [
                            RandomMirrorPoints
                        ]:
                            data[k] = transform(v, rnd_value)
                        else:
                            data[k] = transform(v)

        return data

class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, arr):
        shape = arr.shape
        if len(shape) == 3:    # RGB/Depth Images
            arr = arr.transpose(2, 0, 1)

        # Ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2
        return torch.from_numpy(arr.copy()).float()


class RandomSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:self.n_points]]

        if ptcloud.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])

        return ptcloud


class RandomMirrorPoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
        trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
        if rnd_value <= 0.25:
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
        elif rnd_value > 0.5 and rnd_value <= 0.75:
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud


class NormalizeObjectPose(object):
    def __init__(self, parameters):
        input_keys = parameters['input_keys']
        self.ptcloud_key = input_keys['ptcloud']
        self.bbox_key = input_keys['bbox']

    def __call__(self, data):
        ptcloud = data[self.ptcloud_key]
        bbox = data[self.bbox_key]

        # Calculate center, rotation and scale
        # References:
        # - https://github.com/wentaoyuan/pcn/blob/master/test_kitti.py#L40-L52
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale
        ptcloud = np.dot(ptcloud - center, rotation) / scale
        ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        data[self.ptcloud_key] = ptcloud
        return data


class NormalizePoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, pcd):
        mean_pcd = pcd - np.mean(pcd, axis=0)
        max_coord = np.linalg.norm(mean_pcd, axis=1).max()
        normalized_pcd = mean_pcd / max_coord
        return normalized_pcd


"""
this file contains various functions for point cloud transformation,
some of which are not used in the clean version of code,
but feel free to use them if you have different forms of point clouds.
"""


def swap_axis(input_np, swap_mode='n210'):
    """
    swap axis for point clouds with different canonical frame
    e.g., pcl2pcl and MPC
    """
    if swap_mode == '021':
        output_np = np.stack([input_np[:,0], input_np[:,2], input_np[:,1]],axis=1)
    elif swap_mode == 'n210':
        output_np = np.stack([input_np[:,2]*(-1), input_np[:,1], input_np[:,0]],axis=1)
    elif swap_mode == '210':
        output_np = np.stack([input_np[:,2], input_np[:,1], input_np[:,0]],axis=1)
    else:
        raise NotImplementedError

    return output_np

def scale_numpy(input_array, range=0.25,ax_wise=True):
    """
    scale point cloud in the form of numpy array
    """
    if ax_wise:
        max_abs = np.max(np.abs(input_array),axis=0)
        d0 = input_array[:,0] * (range/max_abs[0])
        d1 = input_array[:,1] * (range/max_abs[1])
        d2 = input_array[:,2] * (range/max_abs[2])
        scaled_array = np.stack([d0,d1,d2], axis=1)
    else:
        """
        scale all dimension by the same value, ie the max(abs)
        """
        max_abs = np.max(np.abs(input_array))
        scaled_array = input_array * (range/max_abs)
    return scaled_array

def scale_numpy_ls(input_ls, range=0.25):
    """
    calling a list of point clouds
    """
    output_ls = []
    for itm in input_ls:
        output = scale_numpy(itm, range=range)
        output_ls.append(output)
    return output_ls

def shift_numpy(input_array, mode='center',additional_limit=None):
    """
    shift
    """
    if mode == 'center':
        ax_max = np.max(input_array,axis=0)
        ax_min = np.min(input_array,axis=0)
        ax_center = (ax_max+ax_min)/2
        shifted_np = input_array - ax_center
    elif mode == 'given_some_limit':
        print(additional_limit)
        if additional_limit[0] != 'yl':
            raise NotImplementedError
        ax_max = np.max(input_array,axis=0)
        ax_min = np.min(input_array,axis=0)
        ax_min[1] = additional_limit[1] # addtional step
        ax_center = (ax_max+ax_min)/2
        shifted_np = input_array - ax_center
    else:
        raise NotImplementedError # weighted center, pc_mean

    return shifted_np

def shift_np_one_dim(input_array, dim=2):
    max_dim = input_array.max(axis=0)
    min_dim = input_array.min(axis=0)
    mean_dim = (max_dim[dim]+min_dim[dim])/2
    input_array[:,dim] -= mean_dim
    return input_array

def downsample_numpy(input_np, points=1024,seed=0):
    if input_np.shape[0] <= points:
        return input_np
    else:
        np.random.seed(seed)
        indices = np.array(range(input_np.shape[0]))
        np.random.shuffle(indices)
        input_downsampled = input_np[indices[:points]]
        return input_downsampled

def voxelize(image, n_bins=32, pcd_limit=0.5, threshold=0):
    """
    voxelize a point cloud
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).unsqueeze(0)
    pcd_new = image * n_bins + n_bins * 0.5
    pcd_new = pcd_new.type(torch.int64)
    ls_voxels = pcd_new.squeeze(0).tolist() # 2028 of sublists
    try:
        tuple_voxels = [tuple(itm) for itm in ls_voxels]
    except:
        import pdb; pdb.set_trace()
    mask_dict = {}
    for tuple_voxel in tuple_voxels:
        if tuple_voxel not in mask_dict:
            mask_dict[tuple_voxel] = 1
        else:
            mask_dict[tuple_voxel] += 1
    for voxel, cnt in mask_dict.items():
        if cnt <= threshold:
            del mask_dict[voxel]
    return mask_dict

def return_plot_range(pcd, plot_range):
    """
    return a range of point cloud,
    to plot Fig.3 in the main paper
    """
    pcd_new = []
    x1, x2 = plot_range[0]
    y1, y2 = plot_range[1]
    z1, z2 = plot_range[2]
    for i in range(2048):
        xs = pcd[i,0]
        ys = pcd[i,2]
        zs = pcd[i,1]
        if x1 < xs < x2 and y1 < ys < y2 and z1 < zs < z2:
            pcd_new.append(pcd[i])
    pcd_new = np.stack(pcd_new)
    return pcd_new

def reverse_normalize(pc, pc_CRN):
    """
    scale up by m and relocate
    """
    m = np.max(np.sqrt(np.sum(pc_CRN**2, axis=1)))
    pc = pc * m
    centroid = np.mean(pc_CRN, axis=0)
    pc = pc + centroid

    return pc

def remove_zeros(partial):
    """
    remove zeros (masked) from a point cloud
    """
    if isinstance(partial, np.ndarray):
        partial = torch.from_numpy(partial)
    norm = torch.norm(partial,dim=1)
    idx =  torch.where(norm > 0)
    partial = partial[idx[0]]
    return partial.numpy()

def retrieve_region(pcd, retrieve_range):
    """
    retrieve a range
    input: np.array (N,3)
    """
    x1, x2 = retrieve_range[0]
    y1, y2 = retrieve_range[1]
    z1, z2 = retrieve_range[2]
    points = []
    for i in range(pcd.shape[0]):
        xs = pcd[i,0]
        ys = pcd[i,2]
        zs = pcd[i,1]
        if x1 < xs < x2 and y1 < ys < y2 and z1 < zs < z2:
            points.append(pcd[i])
    new_pcd = np.stack(points)
    print('new_pcd shape',new_pcd.shape)
    return new_pcd