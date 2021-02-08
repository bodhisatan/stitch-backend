import os
from flask import Flask, request, jsonify
import uuid
from flask_cors import *
import imageio
import cv2
from stitcher import Stitcher
import pic_analysis
import config

app = Flask(__name__)
CORS(app, supports_credentials=True)  # 设置跨域


@app.route('/upload_pic', methods=['post'])
def upload_pic():
    uuid_ = str(uuid.uuid1())
    dir_name = config.nginx_file_url + uuid_
    os.system("mkdir -vp {}".format(dir_name))
    files = request.files.getlist("file")
    file_names = []
    for file in files:
        tmp_pre = str(uuid.uuid1())
        file_names.append(tmp_pre + file.filename)
        file.save(dir_name + "/" + tmp_pre + file.filename)
    return jsonify({'uuid': uuid_, 'pic1_name': file_names[0], 'pic2_name': file_names[1]})


@app.route('/start', methods=['post'])
def start_analysis():
    data = request.get_json()
    dir_name = config.nginx_file_url + data['pic_uuid'] + '/'
    pic1_name = data['pic1_name']
    pic2_name = data['pic2_name']
    pic1_path = dir_name + pic1_name
    pic2_path = dir_name + pic2_name

    image1 = imageio.imread(pic1_path)
    image2 = imageio.imread(pic2_path)
    image1 = cv2.resize(image1, (256, 256))
    image2 = cv2.resize(image2, (256, 256))

    # 拼接图像
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([image1, image2], showMatches=True)
    cv2.imwrite(dir_name + "result.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.imwrite(dir_name + "vis.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # 计算输入图片相似度
    hist = pic_analysis.classify_hist_with_split(image1, image2)
    ssim = pic_analysis.calculate_ssim(image1, image2)

    # 计算psnr
    output = imageio.imread(dir_name + "result.png")
    psnr1 = pic_analysis.psnr(output, image1)
    psnr2 = pic_analysis.psnr(output, image2)
    psnr = round((psnr1 + psnr2) / 2.0, 3)

    # 使用nginx映射本地文件
    return jsonify({'res_url': config.url + data['pic_uuid'] + '/result.png',
                    'vis_url': config.url + data['pic_uuid'] + '/vis.png',
                    'ssim': str(ssim),
                    'hist': str(hist),
                    'psnr': str(psnr)})


if __name__ == '__main__':
    app.run()
