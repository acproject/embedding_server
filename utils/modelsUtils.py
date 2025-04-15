import argparse
from huggingface_hub import snapshot_download
import os

def download_model(model_name, save_path, max_workers=20):
    """
    使用 snapshot_download 下载 Hugging Face 模型到指定路径。
    
    :param model_name: Hugging Face 模型名称，例如 "sentence-transformers/clip-ViT-B-32"
    :param save_path: 保存模型的本地路径
    :param max_workers: 最大并发线程数，默认为 20
    """
    try:
        print(f"正在下载模型 {model_name} 到路径 {save_path}")
        # 确保保存路径存在
        os.makedirs(save_path, exist_ok=True)
        
        # 使用 snapshot_download 下载模型
        snapshot_download(
            repo_id=model_name,
            local_dir=save_path,  # 直接保存到目标路径
            local_dir_use_symlinks=False,  # 禁用符号链接，确保文件直接存储在目标路径
            max_workers=max_workers  # 设置最大并发线程数
        )
        print(f"模型下载完成！已保存至 {save_path}")
    except Exception as e:
        print(f"下载失败: {e}")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="从 Hugging Face 下载模型的工具")
    parser.add_argument("model_name", type=str, help="Hugging Face 模型名称，例如 sentence-transformers/clip-ViT-B-32")
    parser.add_argument("--to", type=str, required=True, help="保存模型的本地路径")
    parser.add_argument("--workers", type=int, default=20, help="最大并发线程数，默认为 20")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用下载函数
    download_model(args.model_name, args.to, args.workers)