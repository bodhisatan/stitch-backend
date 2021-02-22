# -*- coding: utf8 -*-
# author: yaoxianjie
# date: 2021/2/20
from flask import Flask
from flask_cors import CORS
from flask_pymongo import PyMongo
from config import Config


mongo = PyMongo()


def create_app(config_name):
    """
    创建app并进行配置
    :param config_name:
    :return:
    """
    app = Flask(__name__)
    # 配置
    CORS(app, supports_credentials=True)
    mongo.init_app(app, uri=Config.database_url)
    # 创建蓝本
    from app.views.stitch_api import api
    app.register_blueprint(api, url_prefix='/api')

    return app
