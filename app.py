import os
from flask import Flask, request, jsonify
import uuid
from flask_cors import *

app = Flask(__name__)
CORS(app, supports_credentials=True)  # 设置跨域


@app.route('/upload_pic', methods=['post'])
def upload_pic():
    uuid_ = str(uuid.uuid1())
    dir_name = '/Users/bytedance/image/' + uuid_
    os.system("mkdir -vp {}".format(dir_name))
    files = request.files.getlist("file")
    for file in files:
        # print(file)
        file.save(dir_name + "/" + str(uuid.uuid1()) + file.filename)
    return jsonify({'uuid': uuid_})


if __name__ == '__main__':
    app.run()
