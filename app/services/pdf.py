import fitz  # PyMuPDF
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from loguru import logger
import torch
from typing import List, Tuple, Dict, Any, Optional, Union

# 导入PDF布局信息类
from app.services.pdf.pdf_layout_info import PDFLayoutInfo
from app.services.pdf.layout import LayoutAnalyzer
from app.services.pdf.text import TextExtractor
from app.services.pdf.formula import FormulaExtractor
from app.services.pdf.table import TableExtractor
from app.services.pdf.visualization import PDFVisualization

class PDFService:
    # 在PDFService类的__init__方法中，确保正确初始化FormulaExtractor
    def __init__(self, models_dir: str, text_embedder: Any, device: str = "cpu"):
        """
        初始化PDF服务
        
        Args:
            models_dir: 模型目录
            text_embedder: 文本嵌入服务实例
            device: 设备类型，"cpu"或"cuda"
        """
        self.models_dir = models_dir
        self.text_embedder = text_embedder
        self.device = device
        logger.info(f"PDF服务初始化完成，使用设备: {device}")
        
        # 初始化各种分析器
        self.layout_analyzer = LayoutAnalyzer(models_dir=models_dir, device=device)
        self.text_extractor = TextExtractor(models_dir=models_dir, device=device)
        
        # 使用新的Paddle公式识别模型
        logger.info("初始化Paddle公式识别模型...")
        self.formula_extractor = FormulaExtractor(models_dir=models_dir, device=device)
        
        self.table_extractor = TableExtractor(models_dir=models_dir, device=device)
        self.visualizer = PDFVisualization()
        
        # OCR服务将由外部设置
        self.ocr_service = None
        self.ocr_available = False

    def set_ocr_service(self, ocr_service):
        """设置OCR服务"""
        self.ocr_service = ocr_service
        self.ocr_available = True
        logger.info("OCR服务已设置")

    def process_pdf(self, pdf_content: bytes) -> np.ndarray:
        """
        处理PDF并返回嵌入向量
        
        Args:
            pdf_content: PDF文件内容
            
        Returns:
            np.ndarray: 嵌入向量
        """
        # 提取PDF文本
        extracted_text = self._extract_pdf_text(pdf_content)
        
        # 使用文本嵌入服务获取嵌入向量
        embedding = self.text_embedder._get_text_embedding(extracted_text)
        
        return embedding
    
    def analyze_pdf_with_elements(self, pdf_content: bytes, max_pages: int = 24) -> PDFLayoutInfo:
        """
        分析PDF布局并解析公式、表格和图片
        
        Args:
            pdf_content: PDF文档的二进制内容
            max_pages: 最大处理页数
            
        Returns:
            PDFLayoutInfo对象，包含布局和元素信息
        """
        # 初始化布局信息存储结构
        layout_info = PDFLayoutInfo(pdf_content=pdf_content)
        
        # 打开PDF
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        total_pages = len(doc)
        logger.info(f"PDF总页数: {total_pages}")
        
        # 处理所有页面（或最多max_pages页）
        pages_to_process = min(max_pages, total_pages)
        visualization_images = []
        
        for page_idx in range(pages_to_process):
            logger.info(f"\n处理第 {page_idx+1}/{pages_to_process} 页")
            page = doc[page_idx]
            
            # 获取原始页面尺寸
            original_size = (page.rect.width, page.rect.height)
            logger.info(f"原始页面尺寸: {original_size}")
            
            # 渲染页面为图像
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 使用2x缩放获得更清晰的图像
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            img_np = np.array(img)
            
            # 获取渲染后的图像尺寸
            rendered_size = (img.width, img.height)
            logger.info(f"渲染后图像尺寸: {rendered_size}")
            logger.info(f"图像统计信息: 形状={img_np.shape}, 最小值={img_np.min()}, 最大值={img_np.max()}, 平均值={img_np.mean():.2f}")
            
            # 分析布局
            layout_results = self.layout_analyzer.analyze_page(img_np)
            logger.info(f"第 {page_idx+1} 页检测到的布局元素数量: {len(layout_results)}")
            
            # 获取模型处理的尺寸
            model_size = (800, 1024)
            if hasattr(self.layout_analyzer, 'model_input_size'):
                model_size = self.layout_analyzer.model_input_size
            logger.info(f"模型处理尺寸: {model_size}")
            
            # 打印原始布局结果，帮助调试
            logger.info("原始布局结果:")
            for i, block in enumerate(layout_results):
                logger.info(f"  {i+1}. 类型={block['type']}, 坐标={block['coordinates']}, 置信度={block.get('confidence', 0):.2f}")
            
            # 存储布局信息
            layout_info.add_page(page_idx, layout_results, rendered_size, model_size, original_size)
            
            # 创建可视化图像
            orig_img = img.copy()
            layout_img = img.copy()
            draw = ImageDraw.Draw(layout_img)
            
            # 为不同类型的元素使用更鲜艳的颜色
            colors = {
                "plain text": (0, 200, 0),      # 绿色
                "title": (255, 0, 0),           # 红色
                "figure": (0, 0, 255),          # 蓝色
                "table": (255, 165, 0),         # 橙色
                "table_caption": (255, 135, 0), # 深橙色
                "table_footnote": (255, 100, 0),# 红橙色
                "isolate_formula": (128, 0, 128), # 紫色
                "list": (0, 255, 255),          # 青色
                "figure_caption": (0, 20, 45),  # 深蓝色
                "abandon": (100, 100, 100),     # 灰色
                "header": (255, 105, 180),      # 粉色
                "footer": (100, 149, 237)       # 淡蓝色
            }
            
            # 尝试加载更大的字体以提高可读性
            try:
                # 尝试几种常见字体
                for font_name in ["Arial.ttf", "Helvetica.ttf", "DejaVuSans.ttf", "SimHei.ttf"]:
                    try:
                        font = ImageFont.truetype(font_name, 24)
                        break
                    except IOError:
                        continue
                else:
                    font = ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()
            
            logger.info(f"第 {page_idx+1} 页检测到的元素类型:")
            for i, block in enumerate(layout_results):
                block_type = block["type"].lower()  # 转为小写以匹配颜色字典
                logger.info(f"{i+1}. {block_type}")
                
                # 获取模型坐标
                model_coords = block["coordinates"]
                
                # 修正坐标映射方法 - 考虑偏移问题
                x0, y0, x1, y1 = model_coords
                
                # 确保坐标在有效范围内
                x0 = max(0, min(x0, model_size[0]))
                y0 = max(0, min(y0, model_size[1]))
                x1 = max(0, min(x1, model_size[0]))
                y1 = max(0, min(y1, model_size[1]))
                
                # 计算映射比例
                x_ratio = rendered_size[0] / model_size[0]
                y_ratio = rendered_size[1] / model_size[1]
                
                # 应用映射 - 修正偏移问题
                # 根据观察，需要减少一定的偏移量
                offset_x = int(rendered_size[0] * 0.05)  # 水平偏移校正，约5%
                offset_y = int(rendered_size[1] * 0.05)  # 垂直偏移校正，约5%
                
                mapped_x0 = max(0, int(x0 * x_ratio) - offset_x)
                mapped_y0 = max(0, int(y0 * y_ratio) - offset_y)
                mapped_x1 = max(mapped_x0 + 1, min(int(x1 * x_ratio) - offset_x, rendered_size[0]))
                mapped_y1 = max(mapped_y0 + 1, min(int(y1 * y_ratio) - offset_y, rendered_size[1]))
                
                # 打印映射前后的坐标，帮助调试
                logger.info(f"  模型坐标: ({x0}, {y0}, {x1}, {y1}) -> 映射坐标: ({mapped_x0}, {mapped_y0}, {mapped_x1}, {mapped_y1})")
                
                # 更新block中的坐标为映射后的坐标
                block["original_coordinates"] = (mapped_x0, mapped_y0, mapped_x1, mapped_y1)
                
                # 使用映射后的坐标绘制边界框，线宽根据图像大小调整
                line_width = max(3, int(min(rendered_size) / 300))
                color = colors.get(block_type, (0, 0, 200))
                draw.rectangle([mapped_x0, mapped_y0, mapped_x1, mapped_y1], outline=color, width=line_width)
                
                # 添加更清晰的标签
                confidence = block.get("confidence", 0)
                label = f"{i+1}.{block_type} ({confidence:.2f})"
                
                # 为标签添加背景以提高可读性
                text_bbox = draw.textbbox((mapped_x0, mapped_y0-30), label, font=font)
                draw.rectangle(text_bbox, fill=(255, 255, 255, 200))  # 半透明白色背景
                draw.text((mapped_x0, mapped_y0-30), label, fill=color, font=font)
                
                # 尝试提取文本内容
                try:
                    # 确保裁剪区域有效
                    if mapped_x1 > mapped_x0 and mapped_y1 > mapped_y0:
                        # 裁剪区域并提取文本
                        crop_img = img.crop((mapped_x0, mapped_y0, mapped_x1, mapped_y1))
                        
                        # 保存裁剪图像用于调试
                        crop_path = f"/tmp/crop_{page_idx+1}_{i+1}_{block_type}.png"
                        crop_img.save(crop_path)
                        logger.info(f"  裁剪图像已保存到: {crop_path}")
                        
                        # 提取文本
                        # 在处理文本块的部分，使用新的OCR服务
                        # 找到类似这样的代码块：
                        if block_type in ["plain text", "title"]:
                            extracted_text = ""
                            if self.ocr_available and self.ocr_service:
                                # 使用新的OCR服务
                                extracted_text = self.ocr_service.recognize_text(np.array(crop_img))
                            else:
                                # 使用原有的文本提取器作为备选
                                extracted_text = self.text_extractor.extract_text(np.array(crop_img))
                            
                            if extracted_text:
                                # 将提取的文本添加到block中
                                block["extracted_text"] = extracted_text
                                logger.info(f"  文本: {extracted_text[:100]}{'...' if len(extracted_text) > 100 else ''}")
                        
                        # 处理公式
                        elif block_type == "isolate_formula":
                            try:
                                latex = self.formula_extractor.recognize_formula(np.array(crop_img))
                                if latex:
                                    formula_info = {
                                        "element_idx": i,
                                        "coordinates": (mapped_x0, mapped_y0, mapped_x1, mapped_y1),
                                        "latex": latex,
                                        "confidence": confidence
                                    }
                                    layout_info.add_formula(page_idx, formula_info)
                                    block["formula_idx"] = i
                                    logger.info(f"  公式(Paddle识别): {latex[:100]}{'...' if len(latex) > 100 else ''}")
                            except Exception as e:
                                logger.error(f"  公式提取失败: {e}")
                        
                        # 处理表格
                        elif block_type == "table":
                            try:
                                table_markdown = self.table_extractor.extract_table(np.array(crop_img))
                                if table_markdown:
                                    table_info = {
                                        "element_idx": i,
                                        "coordinates": (mapped_x0, mapped_y0, mapped_x1, mapped_y1),
                                        "markdown": table_markdown,
                                        "confidence": confidence
                                    }
                                    layout_info.add_table(page_idx, table_info)
                                    block["table_idx"] = i
                                    logger.info(f"  表格: {table_markdown[:100]}{'...' if len(table_markdown) > 100 else ''}")
                            except Exception as e:
                                logger.error(f"  表格提取失败: {e}")
                        
                        # 处理图片
                        elif block_type == "figure":
                            try:
                                img_base64 = self.visualizer.encode_image_to_base64(crop_img)
                                figure_info = {
                                    "element_idx": i,
                                    "coordinates": (mapped_x0, mapped_y0, mapped_x1, mapped_y1),
                                    "base64": img_base64.split(",")[1] if "," in img_base64 else img_base64,
                                    "confidence": confidence,
                                    "caption": "图片"
                                }
                                layout_info.add_figure(page_idx, figure_info)
                                block["figure_idx"] = i
                                logger.info(f"  图片已提取")
                            except Exception as e:
                                logger.error(f"  图片处理失败: {e}")
                except Exception as e:
                    logger.error(f"  处理失败: {e}")
            
            # 添加原始图像和布局图像到可视化结果
            visualization_images.append(self._encode_image_to_base64(orig_img, f"页面 {page_idx+1} - 原始图像"))
            visualization_images.append(self._encode_image_to_base64(layout_img, f"页面 {page_idx+1} - 布局分析"))
            
            # 显示提取的内容统计
            logger.info("\n提取内容统计:")
            logger.info(f"  文本块: {len([b for b in layout_results if b['type'].lower() in ['plain text', 'title']])}")
            logger.info(f"  公式: {len(layout_info.get_formulas(page_idx))}")
            logger.info(f"  表格: {len(layout_info.get_tables(page_idx))}")
            logger.info(f"  图片: {len(layout_info.get_figures(page_idx))}")
        
        doc.close()
        return layout_info, visualization_images
    
    def process_pdf_with_visualization(self, pdf_content: bytes) -> Tuple[np.ndarray, List[str]]:
        """处理PDF并返回嵌入向量和可视化图像"""
        try:
            # 使用增强的布局分析方法
            layout_info, visualization_images = self.analyze_pdf_with_elements(pdf_content)
            
            # 提取所有文本
            all_texts = []
            for page_idx in range(len(layout_info.pages)):
                page_texts = []
                for block in layout_info.get_page_layout(page_idx):
                    if "extracted_text" in block:
                        page_texts.append(block["extracted_text"])
                all_texts.append("\n".join(page_texts))
            
            # 合并所有页面文本
            combined_text = "\n".join(all_texts)
            
            # 获取文本嵌入向量
            embedding = self.text_embedder._get_text_embedding(combined_text)
            
            return embedding, visualization_images
        except Exception as e:
            logger.error(f"PDF可视化处理失败: {e}")
            raise
    
    def _analyze_layout_with_yolo(self, img: Image.Image, page: fitz.Page, zoom: float) -> Tuple[Image.Image, Dict]:
        draw = ImageDraw.Draw(img)
        layout_info = {}
        
        try:
            # 获取原始图像尺寸
            img_width, img_height = img.size
            logger.info(f"原始图像尺寸: 宽={img_width}, 高={img_height}")

            # 定义YOLO模型期望的输入尺寸 (需要根据实际模型调整)
            # 假设 DocLayout-YOLO 使用 800x1280 或类似的固定尺寸
            # 更新为建议的尺寸，实际应用中应从模型配置或文档获取
            yolo_input_width = 800  # 更新为建议值
            yolo_input_height = 1280 # 更新为建议值
            logger.info(f"假设YOLO输入尺寸: 宽={yolo_input_width}, 高={yolo_input_height}")

            # 计算缩放比例
            scale_x = img_width / yolo_input_width
            scale_y = img_height / yolo_input_height
            logger.info(f"坐标缩放比例: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")
            
            # 这里应该是调用YOLO模型的代码
            # 假设模型输入是调整大小后的图像 (e.g., yolo_input_width x yolo_input_height)
            # 并且模型输出的 bbox 是相对于这个调整后尺寸的归一化坐标 [x_center, y_center, width, height]
            # 例如: predictions = yolo_model.predict(resized_img_np)
            
            # 由于没有实际的YOLO模型，我们模拟一些预测结果
            # 假设这些 bbox 是相对于 yolo_input_width x yolo_input_height 的归一化坐标
            mock_predictions = [
                {"class": "title", "confidence": 0.95, "bbox": [0.5, 0.125, 0.8, 0.05]}, # [x_center_norm, y_center_norm, width_norm, height_norm]
                {"class": "text", "confidence": 0.92, "bbox": [0.5, 0.4, 0.8, 0.4]},
                {"class": "figure", "confidence": 0.88, "bbox": [0.5, 0.75, 0.6, 0.2]}
            ]
            
            logger.info("模拟YOLO预测结果 (归一化到YOLO输入尺寸):")
            for i, pred in enumerate(mock_predictions):
                logger.info(f"  预测 {i+1}: 类别={pred['class']}, 置信度={pred['confidence']:.2f}, 归一化BBox={pred['bbox']}")
            
            # 处理YOLO预测结果
            for pred in mock_predictions:  # 实际使用时替换为真实预测
                class_name = pred["class"]
                confidence = pred["confidence"]
                
                # 获取相对于YOLO输入尺寸的归一化坐标 [x_center, y_center, width, height]
                x_center_norm, y_center_norm, width_norm, height_norm = pred["bbox"]
                
                # 1. 转换为相对于YOLO输入尺寸的绝对像素坐标
                x_center_yolo = x_center_norm * yolo_input_width
                y_center_yolo = y_center_norm * yolo_input_height
                width_yolo = width_norm * yolo_input_width
                height_yolo = height_norm * yolo_input_height
                
                # 2. 使用缩放比例映射回原始图像的绝对像素坐标
                x_center_orig = x_center_yolo * scale_x
                y_center_orig = y_center_yolo * scale_y
                width_orig = width_yolo * scale_x
                height_orig = height_yolo * scale_y
                
                # 3. 转换为原始图像上的左上右下坐标 (x0, y0, x1, y1)
                x0 = int(x_center_orig - width_orig / 2)
                y0 = int(y_center_orig - height_orig / 2)
                x1 = int(x_center_orig + width_orig / 2)
                y1 = int(y_center_orig + height_orig / 2)
                
                logger.info(f"处理 {class_name}: 原始图像坐标 左上=({x0}, {y0}), 右下=({x1}, {y1})")
                
                # 确保坐标在原始图像范围内
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(img_width, x1)
                y1 = min(img_height, y1)
                
                # 根据类别选择不同的颜色
                color_map = {
                    "plain text": (0, 200, 0),      # 绿色
                    "title": (255, 0, 0),           # 红色
                    "figure": (0, 0, 255),          # 蓝色
                    "table": (255, 165, 0),         # 橙色
                    "table_caption": (255, 135, 0), # 深橙色
                    "table_footnote": (255, 100, 0),# 红橙色
                    "isolate_formula": (128, 0, 128), # 紫色
                    "list": (0, 255, 255),          # 青色
                    "figure_caption": (0, 20, 45),  # 深蓝色
                    "abandon": (100, 100, 100),     # 灰色
                    "header": (255, 105, 180),      # 粉色
                    "footer": (100, 149, 237)       # 淡蓝色
                }
                color = color_map.get(class_name, (200, 200, 200))
                
                # 绘制边界框
                draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                
                # 添加标签
                label = f"{class_name} {confidence:.2f}"
                draw.text((x0, y0-15), label, fill=color)
                
                # 保存布局信息
                if class_name not in layout_info:
                    layout_info[class_name] = []
                layout_info[class_name].append({
                    "bbox": [x0, y0, x1, y1],
                    "confidence": confidence
                })
        
        except Exception as e:
            logger.error(f"YOLO布局分析失败: {e}")
            # 在图像上添加错误信息
            draw.text((10, 10), f"YOLO分析失败: {str(e)}", fill=(255, 0, 0))
        
        return img, layout_info

    # Removed duplicate _analyze_layout_with_yolo method
    
    def _analyze_layout_with_pymupdf(self, img: Image.Image, page: fitz.Page, zoom: float) -> Image.Image:
        """使用PyMuPDF原生方法分析页面布局"""
        draw = ImageDraw.Draw(img)
        
        # 获取图像尺寸
        img_width, img_height = img.size
        
        # 1. 绘制文本块
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:  # 文本块
                bbox = block["bbox"]
                # 使用缩放因子转换坐标
                x0, y0, x1, y1 = [coord * zoom for coord in bbox]
                
                # 绘制边界框
                draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
                
                # 检查是否可能是公式
                text = "".join([span["text"] for line in block["lines"] for span in line["spans"]])
                formula_keywords = ["=", "+", "-", "*", "/", "∫", "∑", "∏", "√", "∞", "≈", "≠", "≤", "≥", "α", "β", "γ", "δ", "π"]
                if any(keyword in text for keyword in formula_keywords):
                    draw.rectangle([x0, y0, x1, y1], outline=(128, 0, 128), width=2)
                    draw.text((x0, y0-15), "公式", fill=(128, 0, 128))
        
        # 2. 绘制图像
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            bbox = page.get_image_bbox(xref)
            if bbox:
                x0, y0, x1, y1 = bbox.x0 * zoom, bbox.y0 * zoom, bbox.x1 * zoom, bbox.y1 * zoom
                draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
                draw.text((x0, y0-15), f"图像 {img_idx+1}", fill=(255, 0, 0))
        
        # 3. 绘制表格
        try:
            for i, table in enumerate(page.find_tables()):
                x0, y0, x1, y1 = table.rect.x0 * zoom, table.rect.y0 * zoom, table.rect.x1 * zoom, table.rect.y1 * zoom
                draw.rectangle([x0, y0, x1, y1], outline=(255, 165, 0), width=2)
                draw.text((x0, y0-15), f"表格 {i+1}", fill=(255, 165, 0))
        except Exception as e:
            logger.warning(f"表格检测失败: {e}")
        
        return img
    
    def _extract_pdf_text(self, pdf_content: bytes) -> str:
        """
        从PDF中提取文本
        
        Args:
            pdf_content: PDF文件内容
            
        Returns:
            str: 提取的文本
        """
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            text_parts = []
            
            for page_idx, page in enumerate(doc):
                # 优先使用PDF原生文本提取
                page_text = page.get_text()
                
                # 如果页面文本为空，尝试使用OCR
                if not page_text.strip() and self.ocr_available:
                    # 渲染页面为图像
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    # 使用OCR提取文本
                    import pytesseract
                    page_text = pytesseract.image_to_string(img, lang='chi_sim+eng')
                    logger.info(f"页面 {page_idx+1} 使用OCR提取文本")
                
                text_parts.append(page_text)
            
            # 关闭文档
            doc.close()
            
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"PDF文本提取失败: {e}")
            raise
    
    def _encode_image_to_base64(self, image: Image.Image, title: Optional[str] = None) -> str:
        """
        将PIL图像编码为base64字符串
        
        Args:
            image: PIL图像
            title: 图像标题
            
        Returns:
            str: base64编码的图像
        """
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

img = Image.open(io.BytesIO(img_data))
img_np = np.array(img)
blocks = layout_analyzer.analyze_page(img_np)
viz_img = layout_analyzer.visualize_layout(img_np, blocks)