# stitch-backend

> 图像拼接系统后端，特征提取算法采用SIFT、ORB、Harris和深度学习

## How to start
### 部署前端
* 配置node、Vue环境，参照`https://github.com/bodhisatan/stitch-frontend` 克隆部署前端项目
### 部署数据库
* 安装MongoDB数据库
* 新建数据库存储文件夹，如`D:\mongodbData\data`
* 运行`.\mongod -dbpath D:\mongodbData\data`启动数据库
* 第一次启动时，运行以下命令建表
```bash
$ .\mongo
$ use image-stitch
```
### 部署Nginx
* 安装Nginx
* Nginx配置：
```text
server {
       autoindex on;
       listen       8005;
       server_name  localhost;

       location / {
           root   D:\image-stitch/;
           index  index.html index.htm;
       }
    }
```
* 运行`./nginx` 启动服务器

### 运行后端代码
* 解决依赖
* 按需修改config.py，运行manage.py
