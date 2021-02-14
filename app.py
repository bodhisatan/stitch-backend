import os
from flask import Flask, request, jsonify
import uuid
from flask_cors import *
import imageio
import cv2
from stitcher import Stitcher
import pic_analysis
import config
from PIL import Image
import matplotlib.pyplot as plt
from flask_pymongo import PyMongo

app = Flask(__name__)
CORS(app, supports_credentials=True)  # 设置跨域
app.config['MONGO_URI'] = config.database_url
mongo = PyMongo(app)
collection_name = "stitch-info"


@app.route('/upload_pic', methods=['post'])
def upload_pic():
    uuid_ = str(uuid.uuid1())
    dir_name = config.nginx_file_url + uuid_
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


@app.route('/start', methods=['post'])
def start_analysis():
    data = request.get_json()
    origin_file_name = data["origin_file_name"]
    is_saved = data["isSaved"]
    algorithm = data['algorithm']
    dir_name = config.nginx_file_url + data['pic_uuid'] + '/'
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
        col = mongo.db[collection_name]
        col.insert_one(data_dict)

    # 使用nginx映射本地文件
    return jsonify({'res_url': config.url + data['pic_uuid'] + '/result.png',
                    'vis_url': config.url + data['pic_uuid'] + '/vis.png',
                    'ssim': str(ssim),
                    'hist': str(hist),
                    'psnr': str(psnr),
                    'algorithm_time_cost': algorithm_time_cost,
                    'total_time_cost': total_time_cost})


@app.route('/get_algorithm_time_cost', methods=['get'])
def get_algorithm_time_cost():
    col = mongo.db[collection_name]
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


if __name__ == '__main__':
    app.run()
