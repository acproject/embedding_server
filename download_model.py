import os
from huggingface_hub import snapshot_download
from loguru import logger

# 设置使用镜像站点（如果在中国大陆）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 设置模型名称和下载路径
model_name = "sentence-transformers/LaBSE"
local_dir = "/Users/acproject/workspace/python_projects/embedding_server/models/LaBSE"

# 确保目录存在
os.makedirs(os.path.dirname(local_dir), exist_ok=True)

logger.info(f"开始下载模型 {model_name} 到 {local_dir}")

try:
    # 下载模型
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        revision="main"
    )
    logger.info(f"模型下载完成: {local_dir}")
except Exception as e:
    logger.error(f"模型下载失败: {e}")
    raise