import os
import io
import time
import logging
import traceback
import numpy as np
import torch
import pdfplumber
from typing import List, Dict, Tuple, Optional, Union

# 导入其他模块
from .visualization import PDFVisualization
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
            self.visualization = PDFVisualization()
            
            logger.info("PDF处理服务初始化成功")
        except Exception as e:
            logger.error(f"PDF处理服务初始化失败: {e}")
            raise
    
    def convert_pdf_to_markdown(self, pdf_content: bytes) -> str:
        """
        将PDF文档转换为Markdown格式
        
        Args:
            pdf_content: PDF文档的二进制内容
            
        Returns:
            str: 转换后的Markdown文本
        """
        # 在try块外初始化变量，确保在异常处理中可访问
        markdown_text = ""
        
        try:
            if not isinstance(pdf_content, bytes):
                raise ValueError("pdf_content必须是bytes类型")
                
            # 检查PDF文件头
            if len(pdf_content) < 4 or pdf_content[:4] != b'%PDF':
                logger.error("无效的PDF文件格式")
                return "*无效的PDF文件*"
                
            pdf_file = io.BytesIO(pdf_content)
            
            with pdfplumber.open(pdf_file) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"PDF共有 {total_pages} 页")
                
                for page_num, page in enumerate(pdf.pages):
                    logger.info(f"处理第 {page_num + 1}/{total_pages} 页")
                    page_content = ""  # 每页单独的内容容器
                    
                    # 提取页面图像 (提高分辨率至 300 DPI)
                    img = page.to_image(resolution=300).original
                    img_np = np.array(img)
                    
                    # 分析页面布局 (使用高分辨率图像)
                    layout = self.layout_analyzer.analyze_page(img_np)
                    logger.debug(f"页面 {page_num + 1} 布局分析结果: {layout}") # 添加日志记录布局结果

                    # 如果没有布局模型或没有检测到任何块，将整个页面作为文本处理
                    if not layout:
                        text = self.text_extractor.extract_text(img_np)
                        if text and text.strip():
                            if page_num > 0:
                                page_content += "---\n\n"
                            page_content += f"{text}\n\n"
                        
                        # 将页面内容添加到总内容
                        markdown_text += page_content
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

                        # 处理不同类型的块
                        # 映射YOLO检测的类型到我们的处理类型
                        if block_type == "Text" or block_type == "plain text":
                            # 使用OCR提取文本
                            text = self.text_extractor.extract_text(crop_img_np)
                            if text and text.strip():
                                page_content += f"{text}\n\n"
                            
                        elif block_type == "Title" or block_type == "title":
                            # 使用OCR提取标题文本
                            title_text = self.text_extractor.extract_text(crop_img_np)
                            if title_text and title_text.strip():
                                page_content += f"## {title_text}\n\n"
                            
                        elif block_type == "List":
                            # 使用OCR提取列表文本并添加列表标记
                            list_text = self.text_extractor.extract_text(crop_img_np)
                            if list_text:
                                list_items = list_text.split('\n')
                                formatted_list = '\n'.join([f"- {item}" for item in list_items if item.strip()])
                                if formatted_list:
                                    page_content += f"{formatted_list}\n\n"
                            
                        elif block_type == "Figure":
                            # 将图像转换为base64并嵌入Markdown
                            img_base64 = self.visualization.image_to_base64(crop_img)
                            page_content += f"![图片](data:image/png;base64,{img_base64})\n\n"
                            
                            # 检查图像中是否有文本
                            img_text = self.text_extractor.extract_text(crop_img_np)
                            if img_text and img_text.strip():
                                page_content += f"*图片文本: {img_text}*\n\n"
                            
                        elif block_type == "Table":
                            # 使用表格识别模型提取表格
                            table_markdown = self.table_extractor.extract_table(crop_img_np)
                            if not table_markdown:
                                # 如果高级表格识别失败，回退到基本方法
                                table_markdown = self.table_extractor.extract_table_basic(page, (x1, y1, x2, y2))
                            if table_markdown:
                                page_content += f"{table_markdown}\n\n"
                            
                        elif block_type == "Formula" or block_type == "isolate_formula":
                            # 识别公式为LaTeX
                            latex_formula = self.formula_extractor.recognize_formula(crop_img_np)
                            if latex_formula:
                                # 根据公式类型添加不同的Markdown标记
                                if block.get("inline", False):
                                    page_content += f"${latex_formula}$\n\n"
                                else:
                                    page_content += f"$$\n{latex_formula}\n$$\n\n"
                            else:
                                # 如果公式识别失败，将其作为图像插入
                                img_base64 = self.visualization.image_to_base64(crop_img)
                                page_content += f"![公式](data:image/png;base64,{img_base64})\n\n"
                        
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
                    
                    # 将页面内容添加到总内容
                    markdown_text += page_content
                
                # 对生成的Markdown进行后处理
                markdown_text = self._post_process_markdown(markdown_text)
                
                # 确保返回非空文本
                if not markdown_text.strip():
                    logger.warning("生成的Markdown文本为空，尝试提取基本文本")
                    return self._fallback_extraction(pdf_content)
                
                return markdown_text
                
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
        
        # 合并连续的列表项
        merged = []
        in_list = False
        for line in processed:
            if line.startswith(('- ', '* ', '1. ')):
                if not in_list:
                    merged.append('')  # 列表前添加空行
                    in_list = True
                merged.append(line)
            else:
                if in_list:
                    merged.append('')  # 列表后添加空行
                    in_list = False
                merged.append(line)
        
        return '\n'.join(merged)
    
    def _fallback_extraction(self, pdf_content: bytes) -> str:
        """基本文本提取回退方法"""
        try:
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                return "\n\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception:
            return "*无法提取PDF内容*"
    
    def process_pdf(self, pdf_content):
        """
        处理PDF并返回嵌入向量
        
        Args:
            pdf_content: PDF文件内容
            
        Returns:
            numpy.ndarray: PDF内容的嵌入向量
        """
        # 将PDF转换为Markdown
        markdown_text = self.convert_pdf_to_markdown(pdf_content)
        
        # 获取Markdown文本的嵌入向量
        return self.text_embedder.get_embedding(markdown_text)
    
    def process_pdf_with_visualization(self, pdf_content):
        """
        处理PDF并返回嵌入向量和可视化图像
        
        Args:
            pdf_content: PDF文件内容
            
        Returns:
            tuple: (embedding, visualization_images)
        """
        import fitz
        
        # 初始化可视化图像列表
        visualization_images = []
        
        try:
            # 创建一个临时文件来保存PDF内容
            with fitz.open(stream=pdf_content, filetype="pdf") as doc:
                all_page_texts = []
                
                # 处理每一页
                for page_idx, page in enumerate(doc):
                    logger.info(f"处理PDF第 {page_idx+1}/{len(doc)} 页")
                    
                    # 渲染页面为图像
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("png")
                    img = self.visualization.bytes_to_pil_image(img_data)
                    
                    # 添加原始页面图像到可视化列表
                    visualization_images.append(
                        self.visualization.encode_image_to_base64(img, f"页面 {page_idx+1} - 原始图像")
                    )
                    
                    # 提取页面文本
                    page_text = page.get_text()
                    all_page_texts.append(page_text)
                    
                    # 可视化页面布局
                    layout_img = self.visualization.visualize_page_layout(img, page)
                    visualization_images.append(
                        self.visualization.encode_image_to_base64(layout_img, f"页面 {page_idx+1} - 布局分析")
                    )
                    
                    # 可视化文本块
                    text_blocks_img = self.visualization.visualize_text_blocks(img, page)
                    visualization_images.append(
                        self.visualization.encode_image_to_base64(text_blocks_img, f"页面 {page_idx+1} - 文本块")
                    )
                    
                    # 可视化图像区域
                    images_img = self.visualization.visualize_images(img, page)
                    visualization_images.append(
                        self.visualization.encode_image_to_base64(images_img, f"页面 {page_idx+1} - 图像区域")
                    )
                    
                    # 可视化表格
                    tables_img = self.visualization.visualize_tables(img, page)
                    visualization_images.append(
                        self.visualization.encode_image_to_base64(tables_img, f"页面 {page_idx+1} - 表格")
                    )
                    
                    # 可视化公式
                    formulas_img = self.visualization.visualize_formulas(img, page)
                    visualization_images.append(
                        self.visualization.encode_image_to_base64(formulas_img, f"页面 {page_idx+1} - 公式")
                    )
                
                # 合并所有页面文本
                combined_text = "\n".join(all_page_texts)
                
                # 获取文本嵌入向量
                embedding = self.text_embedder.get_embedding(combined_text)
                
                return embedding, visualization_images
        except Exception as e:
            logger.error(f"PDF可视化处理失败: {e}")
            # 如果可视化处理失败，回退到常规处理
            return self.process_pdf(pdf_content), []