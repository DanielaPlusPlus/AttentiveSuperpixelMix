#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from skimage import segmentation
import copy
import random
from tqdm import tqdm
import os
import numpy as np
import datetime
import torch.nn.functional as F

def OcCaMix(images_in, labels, model, N, n_seg_max,n_seg_min):
    bsz,C,W_img,H_img = images_in.shape
    rand_index = torch.randperm(bsz)
    target_b = labels[rand_index]

    """attentive patches for images[rand_index]"""
    images_mix = copy.deepcopy(images_in)
    images_seg = copy.deepcopy(images_in)
    model.eval()
    with torch.no_grad():
        feat_map4,_ = model(images_in[rand_index])  # (bsz, C, H, W)
        wt = model.fc.weight.data
        num_class, Cf = wt.shape
        _, _, W_feamap, H_feamap = feat_map4.shape
        grid_scale_h = H_img // H_feamap
        grid_scale_w = W_img // W_feamap
        cam_all = []
        for i in range(0, bsz):
            cam = np.dot(wt.cpu(), feat_map4[i].cpu().reshape(Cf, H_feamap * W_feamap))
            cam = F.relu(torch.tensor(cam), inplace=True)
            cam_all.append(cam)
        cam_all = torch.stack(cam_all)
        """element-wise max pooling"""
        cam_all = cam_all.view(bsz, -1, num_class)
        max_pool = nn.MaxPool1d(kernel_size=num_class, stride=1)
        eval_train_map = max_pool(cam_all)
    eval_train_map = eval_train_map.squeeze(dim=2)
    _, map_topN_idx = torch.topk(eval_train_map, N, dim=1, largest=True)
    map_topN_idx_row = (map_topN_idx // W_feamap).cpu().numpy().tolist()
    map_topN_idx_column = (map_topN_idx % H_feamap).cpu().numpy().tolist()
    lam_batch = []
    for i in range(0,bsz):
        x1 = [grid_scale_h * k for k in map_topN_idx_row[i]]
        x2 = [grid_scale_h * (k + 1)-1 for k in map_topN_idx_row[i]]
        y1 = [grid_scale_w * k for k in map_topN_idx_column[i]]
        y2 = [grid_scale_w * (k + 1)-1 for k in map_topN_idx_column[i]]

        """superpixel boundry for images[rand_index]"""
        img_seg = images_seg[rand_index[i]] # (C,W,H)
        img_seg = img_seg.reshape(img_seg.shape[1], img_seg.shape[2], img_seg.shape[0]) #(W,H,C)
        n_seg = random.randint(n_seg_min, n_seg_max)
        segments_img_map = segmentation.slic(img_seg.cpu().numpy(), n_segments=n_seg, compactness=10)

        """attentive superpixel cutmix"""
        img_pixels_idx = []
        x_idx_img, y_idx_img = [], []
        for k in range(0, N):
            atten_pixels_idx = [(x, y) for x in range(x1[k], x2[k]+1) for y in range(y1[k], y2[k]+1)]  # atten pixels idx in original image
            seg_part_label = [segments_img_map[idx] for idx in atten_pixels_idx] # label for the attentive superpixel
            mask_atten = list(set(seg_part_label)) # Remove duplicates
            overlap_tp = []
            x_idx_patch, y_idx_patch = [], []
            for m in range(0, len(mask_atten)):
                x, y = np.where(segments_img_map == mask_atten[m])
                superP_idx = []
                for j in range(0, len(x)):
                    superP_idx.append((x[j], y[j]))
                overlap_idx = list(set(atten_pixels_idx) & set(superP_idx))
                overlap_pct = len(overlap_idx) / len(superP_idx)
                overlap_tp.append(overlap_pct)
                x_idx_patch.append(x)
                y_idx_patch.append(y)

            overlap_most_idx = np.argmax(overlap_tp)

            x_idx_img.extend(x_idx_patch[overlap_most_idx])
            y_idx_img.extend(y_idx_patch[overlap_most_idx])

        for p in range(0,len(x_idx_img)):
            img_pixels_idx.append((x_idx_img[p], y_idx_img[p]))

        img_pixels_idx = list(set(img_pixels_idx))
        nb_mix_pixels = len(img_pixels_idx)

        x_final_idx = [k[0] for k in img_pixels_idx]
        y_final_idx = [k[1] for k in img_pixels_idx]
        images_mix[i, :, x_final_idx, y_final_idx] = \
            images_in[rand_index[i], :,  x_final_idx, y_final_idx]

        lam = nb_mix_pixels / (W_img*H_img)
        lam_batch.append(lam)
    lam_batch = torch.as_tensor(lam_batch).cuda()

    return images_mix, target_b, lam_batch

