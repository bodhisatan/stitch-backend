# -*- coding: utf8 -*-
# author: yaoxianjie
# date: 2021/2/10
import cv2
import scipy
from PIL import Image
from scipy.spatial.distance import cdist
from skimage import filters
from skimage import util
import numpy as np
from skimage.feature import corner_peaks
import matplotlib.pyplot as plt


def gaussian_kernel(size, sigma):
    gaussian_kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = i - (size - 1) / 2
            y = j - (size - 1) / 2
            gaussian_kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return gaussian_kernel


def conv(image, kernel):
    m, n = image.shape
    kernel_m, kernel_n = kernel.shape
    image_pad = np.pad(image, ((kernel_m // 2, kernel_m // 2), (kernel_n // 2, kernel_n // 2)), 'constant')
    result = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            value = np.sum(image_pad[i:i + kernel_m, j:j + kernel_n] * kernel)
            result[i, j] = value
    return result


def harris_corners(image, window_size=3, k=0.04, window_type=0):
    if window_type == 0:
        window = np.ones((window_size, window_size))
    if window_type == 1:
        window = gaussian_kernel(window_size, 1)
    m, n = image.shape
    dx = filters.sobel_v(image)
    dy = filters.sobel_h(image)
    dx_dx = dx * dx
    dy_dy = dy * dy
    dx_dy = dx * dy
    w_dx_dx = conv(dx_dx, window)
    w_dy_dy = conv(dy_dy, window)
    w_dx_dy = conv(dx_dy, window)
    reponse = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            M = np.array([[w_dx_dx[i, j], w_dx_dy[i, j]], [w_dx_dy[i, j], w_dy_dy[i, j]]])
            R = np.linalg.det(M) - k * (np.trace(M)) ** 2
            reponse[i, j] = R
    return reponse


def keypoint_description(image, keypoint, desc_func, patch_size=16):
    keypoint_desc = []
    for i, point in enumerate(keypoint):
        x, y = point
        patch = image[x - patch_size // 2:x + int(np.ceil(patch_size / 2)),
                y - patch_size // 2:y + int(np.ceil(patch_size / 2))]
        description = desc_func(patch)
        keypoint_desc.append(description)
    return np.array(keypoint_desc)


def description_matches(desc1, desc2, threshold=0.5):
    distance_array = cdist(desc1, desc2)
    matches = []
    i = 0
    for each_distance_list in distance_array:
        arg_list = np.argsort(each_distance_list)
        index1 = arg_list[0]
        index2 = arg_list[1]
        if each_distance_list[index1] / each_distance_list[index2] <= threshold:
            matches.append([i, index1])
        i += 1
    return np.array(matches)


def simple_descriptor(patch):
    ave = np.mean(patch)
    std = np.std(patch)
    if std == 0:
        std = 1
    result_patch = (patch - ave) / std
    return result_patch.flatten()


def hog_description(patch, cell_size=(8, 8)):
    if patch.shape[0] % cell_size[0] != 0 or patch.shape[1] % cell_size[1] != 0:
        return 'The size of patch and cell don\'t match'
    n_bins = 9
    degree_per_bins = 20
    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180
    G_as_cells = util.view_as_blocks(G, block_shape=cell_size)
    theta_as_cells = util.view_as_blocks(theta, block_shape=cell_size)
    H = G_as_cells.shape[0]
    W = G_as_cells.shape[1]
    bins_accumulator = np.zeros((H, W, n_bins))
    for i in range(H):
        for j in range(W):
            theta_cell = theta_as_cells[i, j, :, :]
            G_cell = G_as_cells[i, j, :, :]
            for p in range(theta_cell.shape[0]):
                for q in range(theta_cell.shape[1]):
                    theta_value = theta_cell[p, q]
                    G_value = G_cell[p, q]
                    num_bins = int(theta_value // degree_per_bins)
                    k = int(theta_value % degree_per_bins)
                    bins_accumulator[i, j, num_bins % n_bins] += (degree_per_bins - k) / degree_per_bins \
                                                                 * G_value
                    bins_accumulator[i, j, (num_bins + 1) % n_bins] += k / degree_per_bins * G_value
    Hog_list = []
    for x in range(H - 1):
        for y in range(W - 1):
            block_description = bins_accumulator[x:x + 2, y:y + 2]
            block_description = block_description / np.sqrt(np.sum(block_description ** 2))
            Hog_list.append(block_description)
    return np.array(Hog_list).flatten()

def drawMatches(imageA, imageB, kpsA, kpsB, matches):
    # 初始化可视化图片，将A、B图左右连接到一起
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB
    for one_match in matches:
        index1 = one_match[0]
        index2 = one_match[1]
        ptA = kpsA[index1, 1], kpsB[index2, 1]
        ptB = kpsB[index2, 0] + image1.shape[1], kpsA[index1, 0]
        cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    # 返回可视化结果
    return vis


def fit_affine_matrix(p1, p2):
    assert (p1.shape[0] == p2.shape[0]), 'The number of p1 and p2 are different'
    p1 = np.hstack((p1, np.ones((p1.shape[0], 1))))
    p2 = np.hstack((p2, np.ones((p2.shape[0], 1))))
    H = np.linalg.pinv(p2) @ p1
    H[:, 2] = np.array([0, 0, 1])
    return H


def ransac(keypoint1, keypoint2, matches, n_iters=200, threshold=20):
    N = matches.shape[0]
    match_keypoints1 = np.hstack((keypoint1[matches[:, 0]], np.ones((N, 1))))
    match_keypoints2 = np.hstack((keypoint2[matches[:, 1]], np.ones((N, 1))))
    n_samples = int(N * 0.2)
    n_max = 0
    for i in range(n_iters):
        random_index = np.random.choice(N, n_samples, replace=False)
        p1_choice = match_keypoints1[random_index]
        p2_choice = match_keypoints2[random_index]
        H_choice = np.linalg.pinv(p2_choice) @ p1_choice
        H_choice[:, 2] = np.array([0, 0, 1])
        p1_test = match_keypoints2 @ H_choice
        diff = np.sum((match_keypoints1[:, :2] - p1_test[:, :2]) ** 2, axis=1)
        index = np.where(diff <= threshold)[0]
        n_index = index.shape[0]
        if n_index > n_max:
            H = H_choice
            robust_matches = matches[index]
            n_max = n_index
    return H, robust_matches


def get_output_space(image_ref, images, transforms):
    H_ref, W_ref = image_ref.shape
    corner_ref = np.array([[0, 0, 1], [H_ref, 0, 1], [0, W_ref, 1], [H_ref, W_ref, 1]])
    all_corners = [corner_ref]
    if len(images) != len(transforms):
        print('The size of images and transforms does\'t match')
    for i in range(len(images)):
        H, W = images[i].shape
        corner = np.array([[0, 0, 1], [H, 0, 1], [0, W, 1], [H, W, 1]]) @ transforms[i]
        all_corners.append(corner)
    all_corners = np.vstack(all_corners)
    max_corner = np.max(all_corners, axis=0)
    min_corner = np.min(all_corners, axis=0)
    out_space = np.ceil((max_corner - min_corner)[:2]).astype(int)
    offset = min_corner[:2]
    return out_space, offset


def warp_image(image, H, output_shape, offset):
    H_invT = np.linalg.inv(H.T)
    matrix = H_invT[:2, :2]
    o = offset + H_invT[:2, 2]
    image_warped = scipy.ndimage.interpolation.affine_transform(image, matrix, o, output_shape, cval=-1)
    return image_warped

