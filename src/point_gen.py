import cv2
import os
import random
import torch
import numpy as np
import skimage.draw
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F


def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask


def NMS(heatmap, kernel=13):
    hmax = F.max_pool2d(heatmap, kernel, stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heatmap).float()
    return heatmap * keep, hmax, keep


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]], g[g_y[0]:g_y[1],
                                                         g_x[0]:g_x[1]])
    return heatmap


def kpm_gen(label_path, R, N):
    label = np.load(label_path)
    #     label = label[0]
    label_ori = label.copy()
    label = label[::4, ::4]
    label = np.uint8(label * 255)
    contours, hierarchy = cv2.findContours(label, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_NONE)
    contour_len = len(contours)

    label = np.repeat(label[..., np.newaxis], 3, axis=-1)
    draw_label = cv2.drawContours(label.copy(), contours, -1, (0, 0, 255), 1)

    point_file = []
    if contour_len == 0:
        point_heatmap = np.zeros((512, 512))
    else:
        point_heatmap = np.zeros((512, 512))
        for contour in contours:
            stds = []
            points = contour[:, 0]  # (N,2)
            points = points * 4
            points_number = contour.shape[0]
            if points_number < 30:
                continue

            if points_number < 100:
                radius = 6
                neighbor_points_n_oneside = 3
            elif points_number < 200:
                radius = 10
                neighbor_points_n_oneside = 15
            elif points_number < 300:
                radius = 10
                neighbor_points_n_oneside = 20
            elif points_number < 350:
                radius = 15
                neighbor_points_n_oneside = 30
            else:
                radius = 10
                neighbor_points_n_oneside = 40

            for i in range(points_number):
                current_point = points[i]
                mask = create_circular_mask(512, 512, points[i], radius)
                overlap_area = np.sum(
                    mask * label_ori) / (np.pi * radius * radius)
                stds.append(overlap_area)
            print("stds len: ", len(stds))

            # show
            selected_points = []
            stds = np.array(stds)
            neighbor_points = []
            for i in range(len(points)):
                current_point = points[i]
                neighbor_points_index = np.concatenate([
                    np.arange(-neighbor_points_n_oneside, 0),
                    np.arange(1, neighbor_points_n_oneside + 1)
                ]) + i
                neighbor_points_index[np.where(
                    neighbor_points_index < 0)[0]] += len(points)
                neighbor_points_index[np.where(
                    neighbor_points_index > len(points) - 1)[0]] -= len(points)
                if stds[i] < np.min(
                        stds[neighbor_points_index]) or stds[i] > np.max(
                            stds[neighbor_points_index]):
                    #                     print(points[i])
                    point_heatmap = draw_msra_gaussian(
                        point_heatmap, (points[i, 0], points[i, 1]), 5)
                    selected_points.append(points[i])

            print("selected_points num: ", len(selected_points))
            #             print(selected_points)
            maskk = np.zeros((512, 512))
            rr, cc = skimage.draw.polygon(
                np.array(selected_points)[:, 1],
                np.array(selected_points)[:, 0])
            maskk[rr, cc] = 1
            intersection = np.logical_and(label_ori, maskk)
            union = np.logical_or(label_ori, maskk)
            iou_score = np.sum(intersection) / np.sum(union)
            print(iou_score)
    return label_ori, point_heatmap
