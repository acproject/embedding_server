import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
import torch
import os  # 添加 os 模块导入

class EmbeddingService:
    """
    使用sentence-transformers模型将文本转换为向量表示的服务
    """
    
    def __init__(self, model_name="sentence-transformers/LaBSE"):
        """
        初始化嵌入服务
        
        Args:
            model_name: 要使用的模型名称，默认为LaBSE
        """
        try:
            logger.info(f"正在加载模型: {model_name}")  # 使用 logger 而不是 self.logger
            logger.info(f"模型路径是否存在: {os.path.exists(model_name)}")
            
            # 设置设备
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"使用设备: {self.device}")
            
            # 加载模型
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model_name = model_name
            
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def get_embedding(self, text):
        """
        获取文本的嵌入向量
        
        Args:
            text: 要转换的文本
            
        Returns:
            numpy.ndarray: 文本的嵌入向量
        """
        try:
            # 使用模型生成嵌入向量
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"生成嵌入向量时出错: {str(e)}")
            raise
    
    def get_batch_embeddings(self, texts):
        """
        批量获取多个文本的嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            list: 嵌入向量列表
        """
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"批量生成嵌入向量时出错: {str(e)}")
            raise