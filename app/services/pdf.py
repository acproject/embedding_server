import fitz  # PyMuPDF
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from loguru import logger
import torch
from typing import List, Tuple, Dict, Any, Optional, Union

class PDFService:
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
        
        # 尝试导入OCR相关库
        try:
            import pytesseract
            self.ocr_available = True
            logger.info("OCR服务可用")
        except ImportError:
            self.ocr_available = False
            logger.warning("未安装pytesseract，OCR功能不可用")
    
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
    
    def process_pdf_with_visualization(self, pdf_content: bytes) -> Tuple[np.ndarray, List[str]]:
        """处理PDF并返回嵌入向量和可视化图像"""
        visualization_images = []
        
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            all_page_texts = []
            
            for page_idx, page in enumerate(doc):
                logger.info(f"处理PDF第 {page_idx+1}/{len(doc)} 页")
                
                # 使用固定DPI渲染页面为图像
                # 增加DPI以获得更清晰的图像
                dpi = 300
                zoom = dpi / 72  # 72是PDF的默认DPI
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # 将pixmap转换为PIL图像
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # 添加原始页面图像
                visualization_images.append(self._encode_image_to_base64(img, f"页面 {page_idx+1} - 原始图像"))
                
                # 提取页面文本
                page_text = page.get_text()
                all_page_texts.append(page_text)
                
                # 使用PyMuPDF原生方法进行布局分析
                # 注意：这里传递zoom参数，但在方法内部会重新计算实际比例
                native_layout_img = self._analyze_layout_with_pymupdf(img.copy(), page, zoom)
                visualization_images.append(self._encode_image_to_base64(native_layout_img, f"页面 {page_idx+1} - 布局分析"))
                
                # 使用YOLO模型进行布局分析
                layout_img, layout_info = self._analyze_layout_with_yolo(img.copy(), page, zoom)
                visualization_images.append(self._encode_image_to_base64(layout_img, f"页面 {page_idx+1} - YOLO布局分析"))
            
            # 合并所有页面文本
            combined_text = "\n".join(all_page_texts)
            
            # 获取文本嵌入向量
            embedding = self.text_embedder._get_text_embedding(combined_text)
            
            doc.close()
            return embedding, visualization_images
        except Exception as e:
            logger.error(f"PDF可视化处理失败: {e}")
            raise
    
    def _analyze_layout_with_yolo(self, img: Image.Image, page: fitz.Page, zoom: float) -> Tuple[Image.Image, Dict]:
        draw = ImageDraw.Draw(img)
        layout_info = {}
        
        try:
            # 获取图像尺寸
            img_width, img_height = img.size
            logger.info(f"图像尺寸: 宽={img_width}, 高={img_height}")
            
            # 这里应该是调用YOLO模型的代码
            # 例如: predictions = yolo_model.predict(img_np)
            
            # 由于没有实际的YOLO模型，我们模拟一些预测结果
            # 在实际代码中，这部分应该替换为真实的模型预测
            mock_predictions = [
                {"class": "title", "confidence": 0.95, "bbox": [0.1, 0.1, 0.9, 0.15]},
                {"class": "text", "confidence": 0.92, "bbox": [0.1, 0.2, 0.9, 0.6]},
                {"class": "figure", "confidence": 0.88, "bbox": [0.2, 0.65, 0.8, 0.85]}
            ]
            
            # 输出YOLO预测结果的原始格式
            logger.info("YOLO预测结果原始格式:")
            for i, pred in enumerate(mock_predictions):
                logger.info(f"预测 {i+1}: 类别={pred['class']}, 置信度={pred['confidence']}, 边界框={pred['bbox']}")
            
            # 处理YOLO预测结果
            for pred in mock_predictions:  # 实际使用时替换为真实预测
                class_name = pred["class"]
                confidence = pred["confidence"]
                
                # 获取归一化坐标
                x_center, y_center, width, height = pred["bbox"]
                logger.info(f"处理 {class_name} 的归一化坐标: 中心点=({x_center}, {y_center}), 宽高=({width}, {height})")
                
                # 正确的归一化坐标映射
                x_center_abs = x_center * img_width
                y_center_abs = y_center * img_height
                width_abs = width * img_width
                height_abs = height * img_height
                
                logger.info(f"映射到图像像素: 中心点=({x_center_abs}, {y_center_abs}), 宽高=({width_abs}, {height_abs})")
                
                # 转换为左上右下坐标
                x0 = int(x_center_abs - width_abs / 2)
                y0 = int(y_center_abs - height_abs / 2)
                x1 = int(x_center_abs + width_abs / 2)
                y1 = int(y_center_abs + height_abs / 2)
                
                logger.info(f"最终边界框坐标: 左上=({x0}, {y0}), 右下=({x1}, {y1})")
                
                # 确保坐标在图像范围内
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(img_width, x1)
                y1 = min(img_height, y1)
                
                # 根据类别选择不同的颜色
                color_map = {
                    "title": (255, 0, 0),      # 红色
                    "text": (0, 255, 0),       # 绿色
                    "figure": (0, 0, 255),     # 蓝色
                    "table": (255, 165, 0),    # 橙色
                    "formula": (128, 0, 128)   # 紫色
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
    
    def _analyze_layout_with_yolo(self, img: Image.Image, yolo_results, net_size=(640, 640)):
        draw = ImageDraw.Draw(img)
        w_img, h_img = img.size
        w_net, h_net = net_size
    
        for pred in yolo_results:
            # 假设 pred['bbox'] 是归一化到网络输入的 [x_center, y_center, width, height]
            x_center, y_center, width, height = pred['bbox']
            # 还原到网络输入像素
            x_center *= w_net
            y_center *= h_net
            width *= w_net
            height *= h_net
            # 转为左上右下
            x0_net = x_center - width / 2
            y0_net = y_center - height / 2
            x1_net = x_center + width / 2
            y1_net = y_center + height / 2
            # 映射到渲染图片尺寸
            x0_img = x0_net * w_img / w_net
            y0_img = y0_net * h_img / h_net
            x1_img = x1_net * w_img / w_net
            y1_img = y1_net * h_img / h_net
            # 绘制
            draw.rectangle([x0_img, y0_img, x1_img, y1_img], outline=(0, 0, 255), width=2)
            draw.text((x0_img, y0_img-15), pred['class'], fill=(0, 0, 255))
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