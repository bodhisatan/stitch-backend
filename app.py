import os
from flask import Flask, request, jsonify
import uuid
from flask_cors import *

app = Flask(__name__)
CORS(app, supports_credentials=True)  # 设置跨域


@app.route('/upload_pic', methods=['post'])
def upload_pic():
    file = request.files['file']
    filename = file.filename
    dir_name = '/Users/bytedance/image/' + str(uuid.uuid1())
    os.system("mkdir -vp {}".format(dir_name))
    file.save(dir_name + "/" + filename)
    return jsonify()


if __name__ == '__main__':
    app.run()
