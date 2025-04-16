import os
import logging
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class TextExtractor:
    """处理PDF文本提取和OCR"""
    
    def __init__(self, models_dir: str, device: str = "cpu"):
        """
        初始化文本提取模块
        
        Args:
            models_dir: 模型目录
            device: 计算设备 ("cpu" 或 "cuda")
        """
        self.models_dir = models_dir
        self.device = device
        self.ocr = None
        
        try:
            self._init_ocr_models()
        except Exception as e:
            logger.error(f"OCR模型初始化失败: {e}")
            raise
    
    def _init_ocr_models(self):
        """初始化OCR模型"""
        try:
            # 初始化PaddleOCR
            from paddleocr import PaddleOCR
            
            # 根据设备选择适当的GPU设置
            use_gpu = self.device != "cpu"
            
            # 初始化OCR模型，支持中英文识别
            self.ocr = PaddleOCR(
                use_gpu=use_gpu,
                lang="ch",  # 中英文识别
                use_angle_cls=True,  # 使用方向分类器
                show_log=False
            )
            
            logger.info("OCR模型初始化成功")
        except ImportError:
            logger.error("PaddleOCR未安装，无法使用OCR功能")
            raise
        except Exception as e:
            logger.error(f"OCR模型初始化失败: {e}")
            raise
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        从图像中提取文本
        
        Args:
            image: 输入图像的numpy数组
            
        Returns:
            str: 提取的文本
        """
        if self.ocr is None:
            logger.warning("OCR模型未初始化，无法提取文本")
            return ""
        
        try:
            ocr_result = self.ocr.ocr(image, cls=True)
            text_lines = []
            
            # OCR结果格式可能会根据PaddleOCR版本有所不同
            if ocr_result and len(ocr_result) > 0:
                for line in ocr_result[0]:
                    if isinstance(line, list) and len(line) >= 2:
                        text_lines.append(line[1][0])  # 提取文本内容
            
            return "\n".join(text_lines)
        except Exception as e:
            logger.error(f"文本提取失败: {e}")
            return ""
    
    def extract_text_with_layout(self, image: np.ndarray, layout: List[Dict]) -> List[Dict]:
        """
        根据布局信息从图像中提取文本
        
        Args:
            image: 输入图像的numpy数组
            layout: 布局信息列表
            
        Returns:
            List[Dict]: 带有文本内容的布局信息列表
        """
        if self.ocr is None:
            logger.warning("OCR模型未初始化，无法提取文本")
            return layout
        
        try:
            # 复制布局信息，避免修改原始数据
            result = []
            
            for block in layout:
                block_copy = block.copy()
                
                # 只处理文本相关的块
                if block["type"] in ["Text", "Title", "List", "Header", "Footer"]:
                    # 获取块的坐标
                    x1, y1, x2, y2 = block["coordinates"]
                    
                    # 裁剪图像
                    crop_img = image[y1:y2, x1:x2]
                    
                    # 提取文本
                    text = self.extract_text(crop_img)
                    block_copy["text"] = text
                
                result.append(block_copy)
            
            return result
        except Exception as e:
            logger.error(f"带布局的文本提取失败: {e}")
            return layout
    
    def extract_structured_text(self, image: np.ndarray) -> Dict:
        """
        从图像中提取结构化文本（标题、段落、列表等）
        
        Args:
            image: 输入图像的numpy数组
            
        Returns:
            Dict: 结构化文本信息
        """
        try:
            # 使用OCR提取基本文本
            basic_text = self.extract_text(image)
            
            # 尝试识别文本结构
            # 这里可以添加更复杂的文本结构分析逻辑
            
            # 简单的结构化处理：按行分割，尝试识别标题和列表
            lines = basic_text.split('\n')
            structured_text = {
                "title": None,
                "paragraphs": [],
                "lists": []
            }
            
            current_paragraph = []
            current_list = []
            in_list = False
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # 简单的标题检测：第一行且较短
                if i == 0 and len(line) < 50:
                    structured_text["title"] = line
                    continue
                
                # 简单的列表项检测
                if line.startswith(('-', '*', '•')) or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
                    if not in_list:
                        # 结束当前段落
                        if current_paragraph:
                            structured_text["paragraphs"].append(" ".join(current_paragraph))
                            current_paragraph = []
                        in_list = True
                    
                    current_list.append(line)
                else:
                    if in_list:
                        # 结束当前列表
                        structured_text["lists"].append(current_list)
                        current_list = []
                        in_list = False
                    
                    current_paragraph.append(line)
            
            # 处理最后的段落或列表
            if current_paragraph:
                structured_text["paragraphs"].append(" ".join(current_paragraph))
            if current_list:
                structured_text["lists"].append(current_list)
            
            return structured_text
        except Exception as e:
            logger.error(f"结构化文本提取失败: {e}")
            return {"title": None, "paragraphs": [], "lists": []}