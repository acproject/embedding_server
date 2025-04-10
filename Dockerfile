# 使用官方 Python 镜像作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_ENDPOINT="https://hf-mirror.com"

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /app/

# 创建模型目录
RUN mkdir -p /app/models/LaBSE

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 下载模型（如果需要在构建时下载）
# 取消注释下面的行，如果您想在构建镜像时下载模型
# RUN python download_model.py

# 暴露端口
EXPOSE 8086

# 启动命令
CMD ["gunicorn", "-c", "gunicorn_conf.py", "app.main:app"]