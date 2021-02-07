import os
from flask import Flask, request, jsonify
import uuid
from flask_cors import *
import imageio
import cv2
from stitcher import Stitcher

app = Flask(__name__)
CORS(app, supports_credentials=True)  # 设置跨域


@app.route('/upload_pic', methods=['post'])
def upload_pic():
    uuid_ = str(uuid.uuid1())
    dir_name = '/Users/bytedance/image/' + uuid_
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
    dir_name = '/Users/bytedance/image/' + data['pic_uuid'] + '/'
    pic1_name = data['pic1_name']
    pic2_name = data['pic2_name']
    pic1_path = dir_name + pic1_name
    pic2_path = dir_name + pic2_name

    image1 = imageio.imread(pic1_path)
    image2 = imageio.imread(pic2_path)
    image1 = cv2.resize(image1, (256, 256))
    image2 = cv2.resize(image2, (256, 256))

    stitcher = Stitcher()
    (result, vis) = stitcher.stitch([image1, image2], showMatches=True)
    cv2.imwrite(dir_name + "result.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.imwrite(dir_name + "vis.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    return jsonify({'res_url': "http://127.0.0.1:8085/" + data['pic_uuid'] + '/result.png',
                    'vis_url': "http://127.0.0.1:8085/" + data['pic_uuid'] + '/vis.png'})


if __name__ == '__main__':
    app.run()
