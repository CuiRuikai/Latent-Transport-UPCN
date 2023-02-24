import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2 as cv
from tqdm import tqdm

from utils.config import get_config
from models.ebm import EBMUPCN
import tools.builder as builder
from utils.o3d_misc import point_save, point_display, to_point_cloud, to_point_cloud_with_color, o3d_point_save
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics




def main(args):
    config = get_config(args)

    net = builder.model_builder(config.model).cuda()
    builder.load_model(net, args.ckpt_path, logger=None)
    _, data_loader = builder.dataset_builder(args, config.dataset.val)
    net = net.eval()

    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(data_loader) # bs is 1

    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
            for idx, (taxonomy_ids, model_ids, data) in enumerate(data_loader):
                taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
                # load data
                model_id = model_ids[0]
                partial = data[0].cuda()
                gt = data[1].cuda()
                # inference
                ret = net(partial)[0]
                # metric
                _metrics = Metrics.get(ret, gt)
                test_metrics.update(_metrics)
                # save
                folder_name = os.path.join(args.save_path, "{:05d}_{}".format(idx, model_id))
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                point_save(partial, folder_name, 'partial', type='ply')
                point_save(gt, folder_name, 'target', type='ply')
                # estimate confidence
                pred = torch.zeros(args.n_estimation, 2048, 3)
                for i in range(args.n_estimation):
                    pred[i] = net(partial)[0].detach().cpu()

                mu = pred.mean(dim=0)
                std = pred.std(dim=0)
                normalized_std = std / std.max()
                intensity = torch.norm(normalized_std, dim=[1])
                intensity = (intensity - intensity.min())/ (intensity.max() - intensity.min())
                intensity = 255 - np.uint8(255*intensity)
                color_map = cv.applyColorMap(intensity, cv.COLORMAP_JET)/255
                color_map = np.squeeze(color_map)
                points_conf = to_point_cloud_with_color(mu, color_map)
                o3d_point_save(points_conf, folder_name, 'confidence', type='ply')
                pbar.update(1)
                # if idx > 10:
                #     break

    test_metrics.avg()
    print('confidence estimation finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--experiment_path', type=str, default='./experiments/conf')
    parser.add_argument('--n_estimation', type=int, default=64)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--num_workers', type=int, default=1)

    args = parser.parse_args()
    print(args)
    main(args)
