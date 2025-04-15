import os
import base64
import io
from typing import Dict, List, Tuple, Union, Optional, Any
from loguru import logger
import numpy as np
import torch
from PIL import Image
import open_clip
from sentence_transformers import SentenceTransformer
from app.services.pdf_service import PDFService

class MultimodalEmbeddingService:
    """
    多模态嵌入服务，支持文本、PDF和图像的嵌入向量生成
    """
    
    def __init__(self, text_model_name="models/LaBSE", image_model_name="models/vector/CLIP"):
        """
        初始化多模态嵌入服务
        
        Args:
            text_model_name: 文本嵌入模型名称，默认为LaBSE
            image_model_name: 图像嵌入模型名称，默认为CLIP ViT-B-32
        """
        try:
            logger.info("正在初始化多模态嵌入服务")
            
            # 设置设备
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"使用设备: {self.device}")
            
            # 加载文本嵌入模型
            logger.info(f"正在加载文本模型: {text_model_name}")
            self.text_model = SentenceTransformer(text_model_name, device=self.device)
            self.text_model_name = text_model_name
            
            # 加载图像嵌入模型
            logger.info(f"正在加载图像模型: {image_model_name}")
            try:
                # 优先从本地加载模型
                model_path = os.path.join(image_model_name, "pytorch_model.bin")
                if os.path.exists(model_path):
                    logger.info(f"从本地路径加载CLIP模型: {image_model_name}")
                    self.image_model, _, self.image_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
                    self.image_model.load_state_dict(torch.load(model_path))
                else:
                    # 如果本地模型不存在，尝试从预训练模型库下载
                    logger.info("从预训练模型库下载CLIP模型")
                    self.image_model, _, self.image_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
                    
                    # 保存模型到本地
                    if os.path.exists(image_model_name) and os.path.isdir(image_model_name):
                        logger.info(f"保存CLIP模型到本地路径: {image_model_name}")
                        os.makedirs(image_model_name, exist_ok=True)
                        torch.save(self.image_model.state_dict(), os.path.join(image_model_name, "pytorch_model.bin"))
            except Exception as e:
                logger.error(f"加载CLIP模型失败: {e}")
                raise
            self.image_model = self.image_model.to(self.device)
            self.image_model_name = image_model_name
            
            # 初始化PDF服务
            self.pdf_service = PDFService()
            
            logger.info("多模态嵌入服务初始化成功")
        except Exception as e:
            logger.error(f"多模态嵌入服务初始化失败: {e}")
            raise
    
    def get_embedding(self, content: Union[str, bytes], content_type: Optional[str] = None) -> np.ndarray:
        """
        获取内容的嵌入向量，自动判断内容类型
        
        Args:
            content: 要转换的内容，可以是文本字符串、PDF二进制数据或图像二进制数据
            content_type: 内容类型，可选值为"text"、"pdf"、"image"，如果为None则自动判断
            
        Returns:
            numpy.ndarray: 内容的嵌入向量
        """
        try:
            # 如果未指定内容类型，则尝试自动判断
            if content_type is None:
                content_type = self._detect_content_type(content)
            
            logger.info(f"处理内容类型: {content_type}")
            
            if content_type == "text":
                return self._get_text_embedding(content)
            elif content_type == "pdf":
                return self._get_pdf_embedding(content)
            elif content_type == "image":
                return self._get_image_embedding(content)
            else:
                raise ValueError(f"不支持的内容类型: {content_type}")
        except Exception as e:
            logger.error(f"生成嵌入向量时出错: {str(e)}")
            raise
    
    def _detect_content_type(self, content: Union[str, bytes]) -> str:
        """
        自动检测内容类型
        """
        if isinstance(content, str):
            # 检查是否为base64编码的图像
            if content.startswith("data:image") and ";base64," in content:
                return "image"
            return "text"
        elif isinstance(content, bytes):
            # 检查是否为PDF文件
            if content.startswith(b"%PDF"):
                return "pdf"
            # 假设其他二进制内容为图像
            return "image"
        else:
            raise ValueError(f"不支持的内容类型: {type(content)}")
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的嵌入向量
        """
        embedding = self.text_model.encode(text, convert_to_numpy=True)
        return embedding
    
    def _get_pdf_embedding(self, pdf_content: bytes) -> np.ndarray:
        """
        获取PDF文档的嵌入向量
        首先将PDF转换为Markdown，然后获取文本嵌入
        """
        # 确保传入的是bytes类型
        if not isinstance(pdf_content, bytes):
            raise ValueError("pdf_content必须是bytes类型")
            
        # 将PDF转换为Markdown
        markdown_text = self.pdf_service.convert_pdf_to_markdown(pdf_content)
        
        # 获取Markdown文本的嵌入向量
        return self._get_text_embedding(markdown_text)
    
    def _get_image_embedding(self, image_content: Union[str, bytes]) -> np.ndarray:
        """
        获取图像的嵌入向量
        """
        try:
            # 处理不同格式的图像输入
            if isinstance(image_content, str) and image_content.startswith("data:image"):
                # 处理base64编码的图像
                base64_data = image_content.split(";base64,")[1]
                image_bytes = base64.b64decode(base64_data)
                image = Image.open(io.BytesIO(image_bytes))
            elif isinstance(image_content, bytes):
                # 处理二进制图像数据
                image = Image.open(io.BytesIO(image_content))
            else:
                raise ValueError("不支持的图像格式")
            
            # 预处理图像
            image_input = self.image_preprocess(image).unsqueeze(0).to(self.device)
            
            # 获取图像特征
            with torch.no_grad():
                image_features = self.image_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 转换为numpy数组
            return image_features.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"图像嵌入生成失败: {e}")
            raise
    
    def get_batch_embeddings(self, contents: List[Union[str, bytes]], content_types: Optional[List[str]] = None) -> np.ndarray:
        """
        批量获取多个内容的嵌入向量
        
        Args:
            contents: 内容列表
            content_types: 内容类型列表，如果为None则自动判断每个内容的类型
            
        Returns:
            numpy.ndarray: 嵌入向量列表
        """
        try:
            embeddings = []
            
            # 如果未提供内容类型列表，则为每个内容自动检测类型
            if content_types is None:
                content_types = [self._detect_content_type(content) for content in contents]
            
            # 处理每个内容
            for i, (content, content_type) in enumerate(zip(contents, content_types)):
                logger.info(f"处理第 {i+1}/{len(contents)} 个内容，类型: {content_type}")
                embedding = self.get_embedding(content, content_type)
                embeddings.append(embedding)
            
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"批量生成嵌入向量时出错: {str(e)}")
            raise