#!/bin/bash
# 设置环境变量
export HF_ENDPOINT="https://hf-mirror.com"
# 启动服务
gunicorn -c gunicorn_conf.py app.main:app