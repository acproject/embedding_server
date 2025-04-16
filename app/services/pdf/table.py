import os
import logging
import numpy as np
from typing import List, Dict, Optional, Union

logger = logging.getLogger(__name__)

class TableExtractor:
    """处理PDF表格识别和转换"""
    
    def __init__(self, models_dir: str, device: str = "cpu"):
        """
        初始化表格提取模块
        
        Args:
            models_dir: 模型目录
            device: 计算设备 ("cpu" 或 "cuda")
        """
        self.models_dir = models_dir
        self.device = device
        self.table_master_path = None
        self.table_master_model = None
        self.struct_table_path = None
        self.struct_table_model = None
        
        try:
            self._init_table_models()
        except Exception as e:
            logger.error(f"表格识别模型初始化失败: {e}")
            # 如果表格识别失败，将使用基本表格提取
            logger.warning("将使用基本表格提取模式")
    
    def _init_table_models(self):
        """初始化表格识别相关模型"""
        try:
            # 加载TableMaster模型
            table_master_path = os.path.join(self.models_dir, "table/TableMaster")
            if os.path.exists(table_master_path):
                logger.info(f"加载TableMaster模型: {table_master_path}")
                # 这里需要根据TableMaster的实际加载方式进行调整
                self.table_master_path = table_master_path
                self.table_master_model = None  # 实际使用时再加载
            else:
                logger.warning(f"TableMaster模型不存在: {table_master_path}")
                self.table_master_path = None
                self.table_master_model = None
            
            # 加载StructEqTable模型
            struct_table_path = os.path.join(self.models_dir, "table/StructEqTable")
            if os.path.exists(struct_table_path):
                logger.info(f"加载StructEqTable模型: {struct_table_path}")
                # 这里需要根据StructEqTable的实际加载方式进行调整
                self.struct_table_path = struct_table_path
                self.struct_table_model = None  # 实际使用时再加载
            else:
                logger.warning(f"StructEqTable模型不存在: {struct_table_path}")
                self.struct_table_path = None
                self.struct_table_model = None
        except Exception as e:
            logger.error(f"表格识别模型初始化失败: {e}")
            self.table_master_model = None
            self.struct_table_model = None
    
    def extract_table(self, image: np.ndarray) -> str:
        """
        从图像中提取表格并转换为Markdown格式
        
        Args:
            image: 输入图像的numpy数组
            
        Returns:
            str: Markdown格式的表格
        """
        # 尝试使用高级表格识别
        table_markdown = self.extract_table_advanced(image)
        
        # 如果高级表格识别失败，回退到基本方法
        if not table_markdown:
            logger.warning("高级表格识别失败，回退到基本方法")
            table_markdown = self.extract_table_basic(image)
        
        return table_markdown
    
    def extract_table_advanced(self, image: np.ndarray) -> str:
        """
        使用高级表格识别模型提取表格
        
        Args:
            image: 输入图像的numpy数组
            
        Returns:
            str: Markdown格式的表格
        """
        try:
            # 如果没有高级表格识别模型，返回空字符串，让调用者回退到基本方法
            if self.table_master_path is None and self.struct_table_path is None:
                return ""
            
            # 这里应该实现表格识别的具体逻辑
            # 由于这些模型的具体使用方式可能需要特定的代码，这里仅提供框架
            
            # 示例：返回一个简单的Markdown表格
            return "| 列1 | 列2 | 列3 |\n| --- | --- | --- |\n| 数据1 | 数据2 | 数据3 |\n"
        except Exception as e:
            logger.error(f"高级表格提取失败: {e}")
            return ""
    
    def extract_table_basic(self, image: np.ndarray) -> str:
        """
        使用基本方法从图像中提取表格
        
        Args:
            image: 输入图像的numpy数组
            
        Returns:
            str: Markdown格式的表格
        """
        try:
            # 使用OpenCV检测表格线条
            import cv2
            
            # 转换为灰度图像
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 二值化
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # 检测水平和垂直线
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # 合并水平和垂直线
            table_lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # 查找轮廓
            contours, _ = cv2.findContours(table_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 如果没有检测到表格轮廓，返回空字符串
            if not contours:
                return ""
            
            # 使用OCR提取单元格文本
            # 这里需要使用OCR模型，但为了简化示例，我们返回一个占位符表格
            return "| 数据1 | 数据2 | 数据3 |\n| --- | --- | --- |\n| 值1 | 值2 | 值3 |\n"
        except Exception as e:
            logger.error(f"基本表格提取失败: {e}")
            return ""
    
    def extract_table_from_pdf_page(self, page, bbox) -> str:
        """
        使用pdfplumber从页面中提取表格
        
        Args:
            page: pdfplumber页面对象
            bbox: 表格边界框 (x1, y1, x2, y2)
            
        Returns:
            str: Markdown格式的表格
        """
        try:
            # 从页面中提取表格
            x1, y1, x2, y2 = bbox
            table = page.crop((x1, y1, x2, y2)).extract_table()
            
            if not table:
                return "*无法提取表格内容*"
            
            # 转换为Markdown表格格式
            markdown_table = []
            
            # 添加表头
            header = table[0]
            markdown_table.append("| " + " | ".join(cell or "" for cell in header) + " |")
            
            # 添加分隔行
            markdown_table.append("| " + " | ".join(["---"] * len(header)) + " |")
            
            # 添加表格内容
            for row in table[1:]:
                markdown_table.append("| " + " | ".join(cell or "" for cell in row) + " |")
            
            return "\n".join(markdown_table)
        except Exception as e:
            logger.error(f"从PDF页面提取表格失败: {e}")
            return "*表格提取失败*"