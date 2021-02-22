# -*- coding: utf8 -*-
# author: yaoxianjie
# date: 2021/2/20
from app import create_app

app = create_app("development")
app.config.DEBUG = True

if __name__ == '__main__':
    app.run()