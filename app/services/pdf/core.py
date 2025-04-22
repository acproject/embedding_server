import os
import io
import time
import logging
import traceback
import numpy as np
import torch
import pdfplumber
from typing import List, Dict, Tuple, Optional, Union, Any
import base64
from PIL import Image

# 导入其他模块
# 移除visualization导入
from .layout import LayoutAnalyzer
from .text import TextExtractor
from .table import TableExtractor
from .formula import FormulaExtractor

logger = logging.getLogger(__name__)

class PDFService:
    """PDF处理服务，负责协调各个模块处理PDF文档"""
    
    def __init__(self, models_dir: str, text_embedder, device: str = "cpu"):
        """
        初始化PDF处理服务
        
        Args:
            models_dir: 模型目录
            text_embedder: 文本嵌入服务
            device: 计算设备 ("cpu" 或 "cuda")
        """
        self.models_dir = models_dir
        self.text_embedder = text_embedder
        self.device = device
        
        try:
            # 初始化各个模块
            self.layout_analyzer = LayoutAnalyzer(models_dir, device)
            self.text_extractor = TextExtractor(models_dir, device)
            self.table_extractor = TableExtractor(models_dir, device)
            self.formula_extractor = FormulaExtractor(models_dir, device)
            
            logger.info("PDF处理服务初始化成功")
        except Exception as e:
            logger.error(f"PDF处理服务初始化失败: {e}")
            raise
    
    # 添加一个简单的图像转base64函数来替代visualization模块的功能
    def _image_to_base64(self, image):
        """将PIL图像转换为base64编码"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def convert_pdf_to_markdown(self, pdf_content: bytes) -> Tuple[str, Dict]:
        """
        将PDF文档转换为Markdown格式，并返回布局信息
        
        Args:
            pdf_content: PDF文档的二进制内容
            
        Returns:
            Tuple[str, Dict]: 转换后的Markdown文本和布局信息
        """
        # 在try块外初始化变量，确保在异常处理中可访问
        markdown_text = ""
        layout_info = {"pages": [], "formulas": []}
        
        try:
            if not isinstance(pdf_content, bytes):
                raise ValueError("pdf_content必须是bytes类型")
                
            # 检查PDF文件头
            if len(pdf_content) < 4 or pdf_content[:4] != b'%PDF':
                logger.error("无效的PDF文件格式")
                return "*无效的PDF文件*", layout_info
                
            pdf_file = io.BytesIO(pdf_content)
            
            with pdfplumber.open(pdf_file) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"PDF共有 {total_pages} 页")
                
                for page_num, page in enumerate(pdf.pages):
                    logger.info(f"处理第 {page_num + 1}/{total_pages} 页")
                    page_content = ""  # 每页单独的内容容器
                    page_layout = []   # 当前页面的布局信息
                    
                    # 提取页面图像 (提高分辨率至 300 DPI)
                    img = page.to_image(resolution=300).original
                    img_np = np.array(img)
                    
                    # 分析页面布局 (使用高分辨率图像)
                    layout = self.layout_analyzer.analyze_page(img_np)
                    logger.debug(f"页面 {page_num + 1} 布局分析结果: {layout}") # 添加日志记录布局结果

                    # 保存页面布局信息
                    page_info = {
                        "page_num": page_num + 1,
                        "width": img.width,
                        "height": img.height,
                        "elements": []
                    }

                    # 如果没有布局模型或没有检测到任何块，将整个页面作为文本处理
                    if not layout:
                        text = self.text_extractor.extract_text(img_np)
                        if text and text.strip():
                            if page_num > 0:
                                page_content += "---\n\n"
                            page_content += f"{text}\n\n"
                        
                        # 将页面内容添加到总内容
                        markdown_text += page_content
                        
                        # 添加整页文本元素到布局信息
                        if text and text.strip():
                            page_info["elements"].append({
                                "type": "Text",
                                "coordinates": [0, 0, img.width, img.height],
                                "content": text
                            })
                        
                        layout_info["pages"].append(page_info)
                        logger.debug(f"第 {page_num+1} 页基本文本提取完成，当前总字符数: {len(markdown_text)}")
                        continue
                    
                    # 添加页面分隔符
                    if page_num > 0:
                        page_content += "---\n\n"
                    
                    # 处理每个布局元素
                    for block_idx, block in enumerate(layout):
                        block_type = block["type"]
                        x1, y1, x2, y2 = block["coordinates"]
                        logger.debug(f"页面 {page_num + 1}, 块 {block_idx+1}: 类型={block_type}, 原始坐标=({x1}, {y1}, {x2}, {y2})")
                        
                        # 确保坐标有效
                        if x1 >= x2 or y1 >= y2:
                            logger.warning(f"页面 {page_num + 1}, 块 {block_idx+1}: 无效坐标 ({x1}, {y1}, {x2}, {y2})，跳过此块")
                            continue
                            
                        try:
                            crop_img = img.crop((x1, y1, x2, y2))
                            crop_img_np = np.array(crop_img)
                            logger.debug(f"页面 {page_num + 1}, 块 {block_idx+1}: 裁剪区域 ({x1}, {y1}, {x2}, {y2}), 裁剪后图像尺寸: {crop_img.size}")
                        except ValueError as crop_error:
                            logger.error(f"页面 {page_num + 1}, 块 {block_idx+1}: 裁剪失败 ({x1}, {y1}, {x2}, {y2}) - {crop_error}")
                            continue # 跳过处理此块

                        # 创建元素基本信息
                        element_info = {
                            "type": block_type,
                            "coordinates": [x1, y1, x2, y2],
                            "confidence": block.get("confidence", 0.0)
                        }

                        # 处理不同类型的块
                        # 映射YOLO检测的类型到我们的处理类型
                        if block_type == "Text" or block_type == "plain text":
                            # 使用OCR提取文本
                            text = self.text_extractor.extract_text(crop_img_np)
                            if text and text.strip():
                                page_content += f"{text}\n\n"
                                element_info["content"] = text
                            
                        elif block_type == "Title" or block_type == "title":
                            # 使用OCR提取标题文本
                            title_text = self.text_extractor.extract_text(crop_img_np)
                            if title_text and title_text.strip():
                                page_content += f"## {title_text}\n\n"
                                element_info["content"] = title_text
                            
                        elif block_type == "List":
                            # 使用OCR提取列表文本并添加列表标记
                            list_text = self.text_extractor.extract_text(crop_img_np)
                            if list_text:
                                list_items = list_text.split('\n')
                                formatted_list = '\n'.join([f"- {item}" for item in list_items if item.strip()])
                                if formatted_list:
                                    page_content += f"{formatted_list}\n\n"
                                    element_info["content"] = list_text
                                    element_info["items"] = [item for item in list_items if item.strip()]
                            
                        elif block_type == "Figure":
                            # 将图像转换为base64并嵌入Markdown
                            img_base64 = self._image_to_base64(crop_img)
                            page_content += f"![图片](data:image/png;base64,{img_base64})\n\n"
                            element_info["image"] = img_base64
                            
                            # 检查图像中是否有文本
                            img_text = self.text_extractor.extract_text(crop_img_np)
                            if img_text and img_text.strip():
                                page_content += f"*图片文本: {img_text}*\n\n"
                                element_info["text"] = img_text
                            
                            # 尝试识别图像中的公式
                            try:
                                formula_latex = self.formula_extractor.recognize_formula(crop_img_np)
                                if formula_latex and not formula_latex.startswith("iVBOR"):
                                    element_info["formula"] = formula_latex
                                    # 添加到公式列表
                                    layout_info["formulas"].append({
                                        "page": page_num + 1,
                                        "coordinates": [x1, y1, x2, y2],
                                        "latex": formula_latex,
                                        "image": img_base64
                                    })
                            except Exception as formula_error:
                                logger.warning(f"图像公式识别失败: {formula_error}")
                            
                        elif block_type == "Table":
                            # 使用表格识别模型提取表格
                            table_markdown = self.table_extractor.extract_table(crop_img_np)
                            if not table_markdown:
                                # 如果高级表格识别失败，回退到基本方法
                                table_markdown = self.table_extractor.extract_table_basic(page, (x1, y1, x2, y2))
                            if table_markdown:
                                page_content += f"{table_markdown}\n\n"
                                element_info["markdown"] = table_markdown
                            
                        elif block_type == "Formula" or block_type == "isolate_formula":
                            # 识别公式为LaTeX
                            latex_formula = self.formula_extractor.recognize_formula(crop_img_np)
                            if latex_formula:
                                # 根据公式类型添加不同的Markdown标记
                                if block.get("inline", False):
                                    page_content += f"${latex_formula}$\n\n"
                                else:
                                    page_content += f"$$\n{latex_formula}\n$$\n\n"
                                
                                element_info["latex"] = latex_formula
                                
                                # 添加到公式列表
                                layout_info["formulas"].append({
                                    "page": page_num + 1,
                                    "coordinates": [x1, y1, x2, y2],
                                    "latex": latex_formula,
                                    "image": self._image_to_base64(crop_img)
                                })
                            else:
                                # 如果公式识别失败，将其作为图像插入
                                img_base64 = self._image_to_base64(crop_img)
                                page_content += f"![公式](data:image/png;base64,{img_base64})\n\n"
                                element_info["image"] = img_base64
                        
                        elif block_type == "abandon":
                            # 忽略被标记为abandon的块
                            logger.debug(f"页面 {page_num + 1}, 块 {block_idx+1}: 类型={block_type}，已忽略")
                            continue
                        
                        else:
                            # 对于未知类型的块，尝试提取文本
                            logger.warning(f"页面 {page_num + 1}, 块 {block_idx+1}: 未知类型 {block_type}，尝试作为文本处理")
                            text = self.text_extractor.extract_text(crop_img_np)
                            if text and text.strip():
                                page_content += f"{text}\n\n"
                                element_info["content"] = text
                        
                        # 添加元素到页面布局信息
                        page_info["elements"].append(element_info)
                    
                    # 将页面内容添加到总内容
                    markdown_text += page_content
                    
                    # 添加页面布局信息
                    layout_info["pages"].append(page_info)
                
                # 对生成的Markdown进行后处理
                markdown_text = self._post_process_markdown(markdown_text)
                
                # 确保返回非空文本
                if not markdown_text.strip():
                    logger.warning("生成的Markdown文本为空，尝试提取基本文本")
                    basic_text, basic_layout = self._fallback_extraction(pdf_content)
                    return basic_text, basic_layout
                
                return markdown_text, layout_info
                
        except Exception as e:
            logger.error(f"PDF处理失败: {e}")
            # 记录错误并返回基本文本
            return self._fallback_extraction(pdf_content)
    
    def _post_process_markdown(self, markdown: str) -> str:
        """后处理方法，保留有效内容"""
        if not markdown.strip():
            return markdown
            
        # 保留合理的换行结构
        processed = []
        prev_line_empty = False
        for line in markdown.split('\n'):
            stripped_line = line.strip()
            if not stripped_line:
                if not prev_line_empty:
                    processed.append('')
                prev_line_empty = True
            else:
                processed.append(line)
                prev_line_empty = False
                
        return '\n'.join(processed)
    
    def _fallback_extraction(self, pdf_content: bytes) -> Tuple[str, Dict]:
        """基本文本提取方法，用于处理失败时的回退"""
        try:
            # ... existing code ...
            
            # 返回基本文本和空布局信息
            return extracted_text, {"pages": [], "formulas": []}
        except Exception as e:
            logger.error(f"基本文本提取失败: {e}")
            return "*无法提取PDF内容*", {"pages": [], "formulas": []}
    
    def process_pdf(self, pdf_content: bytes) -> np.ndarray:
        """
        处理PDF文档并返回嵌入向量
        
        Args:
            pdf_content: PDF文档的二进制内容
            
        Returns:
            np.ndarray: PDF文档的嵌入向量
        """
        # 转换PDF为Markdown
        markdown_text, _ = self.convert_pdf_to_markdown(pdf_content)
        
        # 使用文本嵌入服务获取嵌入向量
        if markdown_text and markdown_text != "*无法提取PDF内容*":
            embedding = self.text_embedder.get_embedding(markdown_text, "text")
            return embedding
        else:
            # 如果无法提取内容，返回零向量
            return np.zeros(self.text_embedder.get_embedding_dimension())
