# -*- coding: utf8 -*-
# author: yaoxianjie
# date: 2021/2/7
import datetime

from utils.harris import *


class Stitcher:
    """
    图片拼接
    """

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False, feature_algorithm='SIFT'):
        time_step1 = datetime.datetime.now()
        # 获取输入图片，输入两张彩色图片
        (imageB, imageA) = images
        (imageB_tmp, imageA_tmp) = images

        # 如果是Harris算法 转化成灰度图 并归一化
        if feature_algorithm == 'Harris':
            imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            imageA = np.float32(imageA)
            imageB = np.float32(imageB)
            # 归一化
            # dst1 = np.zeros(imageA.shape, dtype=np.float32)
            # cv2.normalize(imageA, dst=dst1, alpha=1.0, beta=0, norm_type=cv2.NORM_INF)
            # dst2 = np.zeros(imageB.shape, dtype=np.float32)
            # cv2.normalize(imageB, dst=dst2, alpha=1.0, beta=0, norm_type=cv2.NORM_INF)
            # imageA = dst1
            # imageB = dst2

        time_step2 = datetime.datetime.now()

        # ##################################特征提取#############################################

        # 检测A、B图片的关键特征点，并计算特征描述子
        kpsA, featuresA = self.detectAndDescribe(imageA, algorithm=feature_algorithm)
        kpsB, featuresB = self.detectAndDescribe(imageB, algorithm=feature_algorithm)

        time_step3 = datetime.datetime.now()

        # ##############################特征匹配 + 图像融合#######################################

        # 匹配两张图片的所有特征点，返回匹配结果
        if feature_algorithm == 'SIFT' or feature_algorithm == 'ORB':
            M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

            # 如果返回结果为空，没有匹配成功的特征点，退出算法
            if M is None:
                return None

            # 否则，提取匹配结果
            # H是3x3视角变换矩阵
            (matches, H, status) = M
            # 将图片A进行视角变换，result是变换后图片
            result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))

            # 将图片B传入result图片最左端
            result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        elif feature_algorithm == 'Harris':
            hog_matches = description_matches(featuresA, featuresB, threshold=0.7)
            H, robust_matches = ransac(kpsA, kpsB, hog_matches, threshold=1)

            output_shape, offset = get_output_space(imageA, [imageB], [H])

            image1_warped = warp_image(imageA, np.eye(3), output_shape, offset)
            image1_mask = (image1_warped != -1)
            image1_warped[~image1_mask] = 0

            image2_warped = warp_image(imageB, H, output_shape, offset)
            image2_mask = (image2_warped != -1)
            image2_warped[~image2_mask] = 0

            merged = image1_warped + image2_warped

            overlap = np.maximum(image1_mask * 1 + image2_mask, 1)
            result = merged / overlap

        time_step4 = datetime.datetime.now()

        total_time_cost = (time_step4 - time_step1).microseconds / 1000
        algorithm_time_cost = (time_step3 - time_step2).microseconds / 1000

        # ##################################输出匹配图片##########################################

        # 检测是否需要显示图片匹配
        if showMatches:
            # 生成匹配图片
            if feature_algorithm == 'SIFT' or feature_algorithm == 'ORB':
                vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            elif feature_algorithm == 'Harris':
                vis = self.drawMatches_(imageA_tmp, imageB_tmp, kpsA, kpsB, robust_matches)
            # 返回结果
            return result, vis, algorithm_time_cost, total_time_cost

        # 返回匹配结果
        return result, algorithm_time_cost, total_time_cost

    def detectAndDescribe(self, image, algorithm='SIFT'):
        if algorithm == 'Harris':
            kps = corner_peaks(harris_corners(image), threshold_rel=0.05, exclude_border=8)
            features = keypoint_description(image, kps, hog_description, patch_size=16)
        else:
            if algorithm == 'SIFT':
                # 建立SIFT生成器
                sift = cv2.SIFT_create()
                # 检测SIFT特征点，并计算描述子
                kps, features = sift.detectAndCompute(image, None)
            elif algorithm == 'ORB':
                # 建立ORB生成器
                orb = cv2.ORB_create()
                # 检测关键点和特征描述
                kps, features = orb.detectAndCompute(image, None)

            # 将结果转换成NumPy数组
            kps = np.float32([kp.pt for kp in kps])

        # 返回特征点集，及对应的描述特征
        return kps, features

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # 建立暴力匹配器
        matcher = cv2.DescriptorMatcher_create("BruteForce")

        # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

        matches = []
        for m in rawMatches:
            # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                # 存储两个点在featuresA, featuresB中的索引值
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # 当筛选后的匹配对大于4时，计算视角变换矩阵
        if len(matches) > 4:
            # 获取匹配对的点坐标
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 计算视角变换矩阵
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # 返回结果
            return (matches, H, status)

        # 如果匹配对小于4时，返回None
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化可视化图片，将A、B图左右连接到一起
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # 返回可视化结果
        return vis

    def drawMatches_(self, imageA, imageB, kpsA, kpsB, matches):
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
            ptB = kpsB[index2, 0] + imageA.shape[1], kpsA[index1, 0]
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # 返回可视化结果
        return vis
