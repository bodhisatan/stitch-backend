# -*- coding: utf8 -*-
# author: yaoxianjie
# date: 2021/2/20
import os
from flask import request, jsonify, Blueprint
import uuid
import cv2

from app import mongo
from utils.stitcher import Stitcher
from utils import pic_analysis
from config import Config
import platform

api = Blueprint('api', __name__)


@api.route('/upload_pic', methods=['post'])
def upload_pic():
    uuid_ = str(uuid.uuid1())
    dir_name = Config.nginx_file_url + uuid_
    # 根据系统新建文件夹
    sysstr = platform.system()
    if sysstr == "Windows":
        os.system("mkdir \"{}\"".format(dir_name))
    else:
        os.system("mkdir -vp {}".format(dir_name))
    files = request.files.getlist("file")
    file_names = []
    origin_file_name = files[0].filename
    for file in files:
        tmp_pre = str(uuid.uuid1())
        file_names.append(tmp_pre + file.filename)
        file.save(dir_name + "/" + tmp_pre + file.filename)
    return jsonify({'uuid': uuid_, 'pic1_name': file_names[0], 'pic2_name': file_names[1],
                    'origin_file_name': origin_file_name})


@api.route('/start', methods=['post'])
def start_analysis():
    data = request.get_json()
    origin_file_name = data["origin_file_name"]
    is_saved = data["isSaved"]
    algorithm = data['algorithm']
    dir_name = Config.nginx_file_url + data['pic_uuid'] + '/'
    pic1_name = data['pic1_name']
    pic2_name = data['pic2_name']
    pic1_path = dir_name + pic1_name
    pic2_path = dir_name + pic2_name

    # 读入图片 -> 转换为3通道 -> 转换格式 -> 放缩尺寸
    image1 = pic_analysis.read_three_channel_pic(pic1_path)
    image2 = pic_analysis.read_three_channel_pic(pic2_path)
    (hA, wA) = image1.shape[:2]
    (hB, wB) = image2.shape[:2]
    image1 = cv2.resize(image1, (max(wA, wB), max(hA, hB)))
    image2 = cv2.resize(image2, (max(wA, wB), max(hA, hB)))

    # 拼接图像
    stitcher = Stitcher()
    (result, vis, algorithm_time_cost, total_time_cost) = stitcher.stitch([image1, image2],
                                                                          showMatches=True, feature_algorithm=algorithm)
    cv2.imwrite(dir_name + "result.png", result)
    cv2.imwrite(dir_name + "vis.png", vis)

    # 计算输入图片相似度
    hist = pic_analysis.classify_hist_with_split(image1, image2)
    ssim = pic_analysis.calculate_ssim(image1, image2)

    # 计算psnr
    output = pic_analysis.read_three_channel_pic(dir_name + "result.png")
    psnr1 = pic_analysis.psnr(output, image1)
    psnr2 = pic_analysis.psnr(output, image2)
    psnr = round((psnr1 + psnr2) / 2.0, 3)

    # 入库
    if is_saved:
        data_dict = {
            "pic_name": origin_file_name,
            "algorithm": algorithm,
            "algorithm_time_cost": algorithm_time_cost,
            "total_time_cost": total_time_cost,
            "ssim": float(ssim),
            "hist": float(hist),
            "psnr": float(psnr)
        }
        col = mongo.db.stitch_info
        col.insert_one(data_dict)

    # 使用nginx映射本地文件
    return jsonify({'res_url': Config.nginx_url + data['pic_uuid'] + '/result.png',
                    'vis_url': Config.nginx_url + data['pic_uuid'] + '/vis.png',
                    'ssim': str(ssim),
                    'hist': str(hist),
                    'psnr': str(psnr),
                    'algorithm_time_cost': algorithm_time_cost,
                    'total_time_cost': total_time_cost})


@api.route('/get_algorithm_time_cost', methods=['get'])
def get_algorithm_time_cost():
    col = mongo.db.stitch_info
    infos = col.find()
    algorithm_xAxisList = []
    algorithm_siftList = []
    algorithm_orbList = []
    algorithm_harrisList = []
    for info in infos:
        if info['algorithm'] == 'Harris':
            algorithm_harrisList.append(info["algorithm_time_cost"])
        elif info['algorithm'] == 'ORB':
            algorithm_orbList.append(info["algorithm_time_cost"])
        elif info['algorithm'] == 'SIFT':
            algorithm_siftList.append(info["algorithm_time_cost"])
    length = max(len(algorithm_orbList), max(len(algorithm_harrisList), len(algorithm_siftList)))
    for i in range(0, length):
        algorithm_xAxisList.append(str(i))
    return jsonify({'algorithm_xAxisList': algorithm_xAxisList,
                    'algorithm_siftList': algorithm_siftList,
                    'algorithm_orbList': algorithm_orbList,
                    'algorithm_harrisList': algorithm_harrisList})


@api.route('/get_total_time_cost', methods=['get'])
def get_total_time_cost():
    col = mongo.db.stitch_info
    infos = col.find()
    total_xAxisList = []
    total_siftList = []
    total_orbList = []
    total_harrisList = []
    for info in infos:
        if info['algorithm'] == 'Harris':
            total_harrisList.append(info["total_time_cost"])
        elif info['algorithm'] == 'ORB':
            total_orbList.append(info["total_time_cost"])
        elif info['algorithm'] == 'SIFT':
            total_siftList.append(info["total_time_cost"])
    length = max(len(total_orbList), max(len(total_harrisList), len(total_siftList)))
    for i in range(0, length):
        total_xAxisList.append(str(i))
    return jsonify({'total_xAxisList': total_xAxisList,
                    'total_siftList': total_siftList,
                    'total_orbList': total_orbList,
                    'total_harrisList': total_harrisList})


@api.route('/get_picture_names', methods=['get'])
def get_pic_names():
    col = mongo.db.stitch_info
    infos = col.find()
    name_set = set()
    for info in infos:
        name_set.add(info["pic_name"])
    name_list = list(name_set)
    return jsonify({'name_list': name_list})


# 根据pic_name获取不同算法ssim、hist、psnr、耗时数据
'''
格式：
datas = [
        {
          "category": 'SSIM',
          "algorithmdata": [
            {"algorithmname": 'Harris', "_data": '12.13'},
            {"algorithmname": 'ORB', "_data": '10.3'},
            {"algorithmname": 'SIFT', "_data": '15.5'}
          ]
        },
        {
          "category": '三通道相似度',
          "algorithmdata": [
            {"algorithmname": 'Harris', "_data": '12.13'},
            {"algorithmname": 'ORB', "_data": '10.3'},
            {"algorithmname": 'SIFT', "_data": '15.5'}
          ]
        },
        ...
      ]
'''


@api.route('/get_compare_data', methods=['post'])
def get_compare_data():
    ssim_algorithm_data = []
    hist_algorithm_data = []
    psnr_algorithm_data = []
    feature_time_algorithm_data = []
    tot_time_algorithm_data = []

    data = request.get_json()
    pic_name = data['pic_name']
    col = mongo.db.stitch_info
    infos = col.find()
    for info in infos:
        if info['pic_name'] == pic_name:
            ssim_algorithm_data.append({
                "algorithmname": info['algorithm'],
                "_data": str(info['ssim'])
            })
            hist_algorithm_data.append({
                "algorithmname": info['algorithm'],
                "_data": str(info['hist'])
            })
            psnr_algorithm_data.append({
                "algorithmname": info['algorithm'],
                "_data": str(info['psnr'])
            })
            feature_time_algorithm_data.append({
                "algorithmname": info['algorithm'],
                "_data": info['algorithm_time_cost']
            })
            tot_time_algorithm_data.append({
                "algorithmname": info['algorithm'],
                "_data": info['total_time_cost']
            })
    datas = [
        {
            "category": 'SSIM',
            "algorithmdata": ssim_algorithm_data
        },
        {
            "category": '三通道相似度',
            "algorithmdata": hist_algorithm_data
        },
        {
            "category": 'PSNR',
            "algorithmdata": psnr_algorithm_data
        },
        {
            "category": '特征提取耗时',
            "algorithmdata": feature_time_algorithm_data
        },
        {
            "category": '总耗时',
            "algorithmdata": tot_time_algorithm_data
        },
    ]
    return jsonify({'compare_data': datas})
