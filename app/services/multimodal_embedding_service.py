# 在文件顶部确保导入了这些库
import os
import base64
import io
from typing import Dict, List, Tuple, Union, Optional, Any
from loguru import logger
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import open_clip
from sentence_transformers import SentenceTransformer
from app.services.pdf_service import PDFService

# 添加必要的导入
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64
from PIL import Image
from typing import Tuple, List, Optional, Dict, Any, Union

class MultimodalEmbeddingService:
    """
    多模态嵌入服务，支持文本、PDF和图像的嵌入向量生成
    """
    
    def __init__(self, text_model_name="models/LaBSE", image_model_name="models/vector/CLIP", init_pdf_service=True):
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
            
            # 初始化OCR模型
            try:
                logger.info("正在加载新的OCR模型...")
                from app.services.ocr import OCRService
                self.ocr_service = OCRService(models_dir="./models", device=self.device)
                self.ocr_available = True
                logger.info("新OCR模型加载成功")
            except Exception as e:
                logger.error(f"OCR模型加载失败: {e}")
                self.ocr_available = False
            
            # 初始化PDF服务
            # 只有当init_pdf_service为True时才初始化PDFService
            if init_pdf_service:
                try:
                    from app.services.pdf import PDFService
                    self.pdf_service = PDFService(models_dir="./models", text_embedder=self)
                    # 将OCR服务传递给PDF服务
                    if hasattr(self, 'ocr_service') and self.ocr_available:
                        self.pdf_service.set_ocr_service(self.ocr_service)
                except Exception as e:
                    logger.error(f"PDF服务初始化失败: {e}")
                    self.pdf_service = None
                else:
                    self.pdf_service = None
                
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
            content_types: 内容类型列表，如果为None则自动检测每个内容的类型
            
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

    # 删除这两个重复的get_embedding方法，只保留一个正确的实现
    def get_embedding(self, content: Union[str, bytes], content_type: Optional[str] = None, visualize: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        """
        获取内容的嵌入向量，自动判断内容类型
        
        Args:
            content: 要转换的内容，可以是文本字符串、PDF二进制数据或图像二进制数据
            content_type: 内容类型，可选值为"text"、"pdf"、"image"，如果为None则自动判断
            visualize: 是否启用可视化（仅对PDF有效）
            
        Returns:
            如果content_type为"pdf"且visualize=True，返回(embedding, visualization_images)元组
            否则返回embedding
        """
        try:
            # 如果未指定内容类型，则尝试自动判断
            if content_type is None:
                content_type = self._detect_content_type(content)
            
            logger.info(f"处理内容类型: {content_type}")
            
            if content_type == "text":
                return self._get_text_embedding(content)
            elif content_type == "pdf":
                # 如果启用了可视化，则调用支持可视化的PDF处理方法
                if visualize:
                    if hasattr(self, 'pdf_service') and self.pdf_service:
                        return self.pdf_service.process_pdf_with_visualization(content)
                    else:
                        # 如果PDF服务不可用，则使用内部方法
                        return self._get_pdf_embedding_with_visualization(content)
                else:
                    # 否则使用常规PDF处理方法
                    if hasattr(self, 'pdf_service') and self.pdf_service:
                        return self.pdf_service.process_pdf(content)
                    else:
                        return self._get_pdf_embedding(content)
            elif content_type == "image":
                return self._get_image_embedding(content)
            else:
                raise ValueError(f"不支持的内容类型: {content_type}")
        except Exception as e:
            logger.error(f"生成嵌入向量时出错: {str(e)}")
            raise
    
    # 删除重复的_encode_image_to_base64方法，只保留一个
    def _encode_image_to_base64(self, image, title=None):
        """将PIL图像编码为base64字符串"""
        # 如果提供了标题，在图像上添加标题
        if title:
            draw = ImageDraw.Draw(image)
            # 尝试加载字体，如果失败则使用默认字体
            try:
                font = ImageFont.truetype("Arial", 24)
            except IOError:
                font = ImageFont.load_default()
            
            # 在图像顶部添加标题
            draw.rectangle((0, 0, image.width, 40), fill=(0, 0, 0))
            draw.text((10, 10), title, fill=(255, 255, 255), font=font)
        
        # 将图像编码为base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

    def _visualize_page_layout(self, image, page):
        """可视化页面布局"""
        # 创建图像副本以进行绘制
        viz_img = image.copy()
        draw = ImageDraw.Draw(viz_img)
        
        # 绘制页面边界
        draw.rectangle((0, 0, viz_img.width-1, viz_img.height-1), outline=(255, 0, 0), width=2)
        
        # 获取页面布局信息
        blocks = page.get_text("blocks")
        
        # 为不同类型的块使用不同颜色
        colors = {
            0: (255, 0, 0, 64),    # 文本块 - 红色
            1: (0, 255, 0, 64),    # 图像块 - 绿色
            2: (0, 0, 255, 64),    # 表格块 - 蓝色
            3: (255, 255, 0, 64),  # 公式块 - 黄色
        }
        
        # 绘制每个块
        for block in blocks:
            block_type = block[6]  # 块类型
            x0, y0, x1, y1 = block[:4]  # 块坐标
            
            # 获取块颜色
            color = colors.get(block_type, (128, 128, 128, 64))
            
            # 绘制块边界和填充
            draw.rectangle((x0, y0, x1, y1), outline=color[:3], width=2)
            draw.rectangle((x0, y0, x1, y1), fill=color)
            
            # 添加块类型标签
            block_types = {0: "文本", 1: "图像", 2: "表格", 3: "公式"}
            label = block_types.get(block_type, f"类型{block_type}")
            draw.text((x0, y0-15), label, fill=(0, 0, 0))
        
        return viz_img

    def _visualize_text_blocks(self, image, page):
        """可视化文本块"""
        viz_img = image.copy()
        draw = ImageDraw.Draw(viz_img)
        
        # 获取文本块
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0]  # 类型0是文本块
        
        # 绘制每个文本块
        for block in text_blocks:
            x0, y0, x1, y1 = block[:4]
            text = block[4]
            
            # 绘制文本块边界
            draw.rectangle((x0, y0, x1, y1), outline=(0, 0, 255), width=2)
            
            # 显示部分文本
            display_text = text[:20] + "..." if len(text) > 20 else text
            draw.text((x0, y0-15), display_text, fill=(0, 0, 0))
        
        return viz_img

    def _visualize_images(self, image, page):
        """可视化图像区域"""
        viz_img = image.copy()
        draw = ImageDraw.Draw(viz_img)
        
        # 获取图像块
        blocks = page.get_text("blocks")
        image_blocks = [b for b in blocks if b[6] == 1]  # 类型1是图像块
        
        # 绘制每个图像块
        for i, block in enumerate(image_blocks):
            x0, y0, x1, y1 = block[:4]
            
            # 绘制图像块边界
            draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0), width=2)
            draw.text((x0, y0-15), f"图像 {i+1}", fill=(0, 0, 0))
        
        return viz_img

    def _visualize_tables(self, image, page):
        """可视化表格"""
        viz_img = image.copy()
        draw = ImageDraw.Draw(viz_img)
        
        # 获取表格块
        blocks = page.get_text("blocks")
        table_blocks = [b for b in blocks if b[6] == 2]  # 类型2是表格块
        
        # 绘制每个表格块
        for i, block in enumerate(table_blocks):
            x0, y0, x1, y1 = block[:4]
            
            # 绘制表格块边界
            draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 255), width=2)
            draw.text((x0, y0-15), f"表格 {i+1}", fill=(0, 0, 0))
        
        return viz_img

    def _visualize_formulas(self, image, page):
        """可视化公式"""
        viz_img = image.copy()
        draw = ImageDraw.Draw(viz_img)
        
        # 获取公式块
        blocks = page.get_text("blocks")
        formula_blocks = [b for b in blocks if b[6] == 3]  # 类型3是公式块
        
        # 绘制每个公式块
        for i, block in enumerate(formula_blocks):
            x0, y0, x1, y1 = block[:4]
            formula_text = block[4]
            
            # 绘制公式块边界
            draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 0), width=2)
            
            # 显示公式文本（这里可以看到是否所有公式都被识别为e=mc2）
            draw.text((x0, y0-15), f"公式 {i+1}: {formula_text[:30]}", fill=(0, 0, 0))
        
        return viz_img

    def _encode_image_to_base64(self, image):
        """将PIL图像编码为base64字符串"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"