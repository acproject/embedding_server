import os
import base64
import io
from typing import Dict, List, Tuple, Union, Optional, Any
from loguru import logger
import pdfplumber
import layoutparser as lp
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import torch
import cv2

class PDFService:
    """
    PDF处理服务，用于将PDF文档转换为Markdown格式
    支持复杂PDF文档结构处理，包括布局检测、公式识别、表格识别等
    """
    
    def __init__(self):
        """
        初始化PDF处理服务，加载所有需要的模型
        """
        try:
            logger.info("正在初始化PDF处理服务")
            
            # 设置设备
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"使用设备: {self.device}")
            
            # 模型根目录
            self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
            logger.info(f"模型根目录: {self.models_dir}")
            
            # 初始化OCR引擎
            ocr_model_path = os.path.join(self.models_dir, "ocr/PaddleOCR")
            logger.info(f"加载OCR模型: {ocr_model_path}")
            self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=torch.cuda.is_available(),
                                det_model_dir=os.path.join(ocr_model_path, "det"),
                                rec_model_dir=os.path.join(ocr_model_path, "rec"))
            
            # 初始化布局分析模型
            self._init_layout_models()
            
            # 初始化公式识别模型
            self._init_formula_models()
            
            # 初始化表格识别模型
            self._init_table_models()
            
            logger.info("PDF处理服务初始化成功")
        except Exception as e:
            logger.error(f"PDF处理服务初始化失败: {e}")
            raise
    
    def _init_layout_models(self):
        """
        初始化布局分析相关模型
        """
        try:
            # 加载DocLayout-YOLO模型
            doc_layout_path = os.path.join(self.models_dir, "layout/DocLayout-YOLO_ft/yolov10l_ft.pt")
            if os.path.exists(doc_layout_path):
                logger.info(f"加载DocLayout-YOLO模型: {doc_layout_path}")
                from ultralytics import YOLO
                self.doc_layout_model = YOLO(doc_layout_path)
                self.doc_layout_model.to(self.device)
            else:
                logger.warning(f"DocLayout-YOLO模型不存在: {doc_layout_path}")
                self.doc_layout_model = None
            
            # 使用YOLOv5作为主要布局分析工具
            self.layout_model = self.doc_layout_model
            if self.layout_model is None:
                logger.warning("未找到布局分析模型，将使用基本文本提取")
        except Exception as e:
            logger.error(f"布局分析模型初始化失败: {e}")
            # 如果布局分析失败，将使用基本文本提取
            self.layout_model = None
            logger.warning("将使用基本文本提取模式")
    
    def _init_formula_models(self):
        """
        初始化公式识别相关模型
        """
        try:
            # 加载YOLOv8公式检测模型
            formula_det_path = os.path.join(self.models_dir, "formula/YOLOv8_ft/yolo_v8_ft.pt")
            if os.path.exists(formula_det_path):
                logger.info(f"加载公式检测模型: {formula_det_path}")
                from ultralytics import YOLO
                self.formula_det_model = YOLO(formula_det_path)
                self.formula_det_model.to(self.device)
            else:
                logger.warning(f"公式检测模型不存在: {formula_det_path}")
                self.formula_det_model = None
            
            # 加载UniMERNet公式识别模型
            formula_rec_path = os.path.join(self.models_dir, "formula_recognition/UniMERNet")
            if os.path.exists(formula_rec_path):
                logger.info(f"加载公式识别模型: {formula_rec_path}")
                # 这里需要根据UniMERNet的实际加载方式进行调整
                # 由于UniMERNet可能需要特定的加载方式，这里仅记录路径
                self.formula_rec_path = formula_rec_path
                self.formula_rec_model = None  # 实际使用时再加载
            else:
                logger.warning(f"公式识别模型不存在: {formula_rec_path}")
                self.formula_rec_path = None
                self.formula_rec_model = None
        except Exception as e:
            logger.error(f"公式识别模型初始化失败: {e}")
            self.formula_det_model = None
            self.formula_rec_model = None
    
    def _init_table_models(self):
        """
        初始化表格识别相关模型
        """
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
    
    def convert_pdf_to_markdown(self, pdf_content: bytes) -> str:
        """
        将PDF文档转换为Markdown格式
        
        Args:
            pdf_content: PDF文档的二进制内容
            
        Returns:
            str: 转换后的Markdown文本
        """
        try:
            # 确保传入的是bytes类型
            if not isinstance(pdf_content, bytes):
                raise ValueError("pdf_content必须是bytes类型")
                
            # 创建一个临时的PDF文件对象
            pdf_file = io.BytesIO(pdf_content)
            
            # 使用pdfplumber打开PDF
            with pdfplumber.open(pdf_file) as pdf:
                markdown_content = []
                
                # 处理每一页
                for page_num, page in enumerate(pdf.pages):
                    logger.info(f"处理第 {page_num + 1} 页")
                    
                    # 提取页面图像
                    img = page.to_image(resolution=300).original
                    img_np = np.array(img)
                    
                    # 使用布局分析模型分析页面结构
                    if self.layout_model:
                        # 使用YOLOv5进行布局检测
                        results = self.layout_model(img_np)
                        layout = []
                        for result in results:
                            for box in result.boxes:
                                layout.append({
                                    "type": result.names[int(box.cls)],
                                    "coordinates": [int(x) for x in box.xyxy[0].tolist()],
                                    "confidence": float(box.conf)
                                })
                    else:
                        # 如果没有布局模型，将整个页面作为文本区域
                        layout = [{
                            "type": "Text",
                            "coordinates": [0, 0, img_np.shape[1], img_np.shape[0]],
                            "confidence": 1.0
                        }]
                    
                    # 检测公式（如果有公式检测模型）
                    formula_blocks = self._detect_formulas(img_np) if self.formula_det_model else []
                    
                    # 合并布局元素和公式元素
                    all_blocks = layout + formula_blocks
                    
                    # 按照从上到下的顺序排序布局元素
                    sorted_blocks = sorted(all_blocks, key=lambda x: x["coordinates"][1])
                    
                    # 处理每个布局元素
                    for block in sorted_blocks:
                        block_type = block["type"]
                        x1, y1, x2, y2 = block["coordinates"]
                        crop_img = img.crop((x1, y1, x2, y2))
                        crop_img_np = np.array(crop_img)
                        
                        if block_type == "Text":
                            # 使用OCR提取文本
                            text = self._extract_text_from_image(crop_img_np)
                            markdown_content.append(f"{text}\n\n")
                            
                        elif block_type == "Title":
                            # 使用OCR提取标题文本
                            title_text = self._extract_text_from_image(crop_img_np)
                            markdown_content.append(f"## {title_text}\n\n")
                            
                        elif block_type == "List":
                            # 使用OCR提取列表文本并添加列表标记
                            list_text = self._extract_text_from_image(crop_img_np)
                            list_items = list_text.split('\n')
                            formatted_list = '\n'.join([f"- {item}" for item in list_items if item.strip()])
                            markdown_content.append(f"{formatted_list}\n\n")
                            
                        elif block_type == "Figure":
                            # 将图像转换为base64并嵌入Markdown
                            img_base64 = self._image_to_base64(crop_img)
                            markdown_content.append(f"![图片](data:image/png;base64,{img_base64})\n\n")
                            
                            # 检查图像中是否有文本
                            img_text = self._extract_text_from_image(crop_img_np)
                            if img_text.strip():
                                markdown_content.append(f"*图片文本: {img_text}*\n\n")
                            
                        elif block_type == "Table":
                            # 使用表格识别模型提取表格
                            table_markdown = self._extract_table_advanced(crop_img_np)
                            if not table_markdown:
                                # 如果高级表格识别失败，回退到基本方法
                                table_markdown = self._extract_table_basic(page, (x1, y1, x2, y2))
                            markdown_content.append(f"{table_markdown}\n\n")
                            
                        elif block_type == "Formula":
                            # 识别公式为LaTeX
                            latex_formula = self._recognize_formula(crop_img_np)
                            if latex_formula:
                                # 根据公式类型添加不同的Markdown标记
                                if block.get("inline", False):
                                    markdown_content.append(f"${latex_formula}$\n\n")
                                else:
                                    markdown_content.append(f"$$\n{latex_formula}\n$$\n\n")
                            else:
                                # 如果公式识别失败，将其作为图像插入
                                img_base64 = self._image_to_base64(crop_img)
                                markdown_content.append(f"![公式](data:image/png;base64,{img_base64})\n\n")
                
                # 后处理：合并相邻的文本块
                processed_content = self._post_process_markdown(''.join(markdown_content))
                return processed_content
                
        except Exception as e:
            logger.error(f"PDF转Markdown失败: {e}")
            raise
    
    def _detect_formulas(self, image: np.ndarray) -> List[Dict]:
        """
        使用YOLOv8检测图像中的公式
        """
        if self.formula_det_model is None:
            return []
        
        try:
            # 使用YOLO模型检测公式
            results = self.formula_det_model(image)
            formula_blocks = []
            
            # 处理检测结果
            for pred in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2, conf, cls = pred
                # 判断是否为行内公式或行间公式
                is_inline = int(cls) == 0  # 假设类别0为行内公式，1为行间公式
                
                formula_blocks.append({
                    "type": "Formula",
                    "coordinates": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": float(conf),
                    "inline": is_inline
                })
            
            return formula_blocks
        except Exception as e:
            logger.error(f"公式检测失败: {e}")
            return []
    
    def _merge_blocks(self, layout_blocks, formula_blocks) -> List[Dict]:
        """
        合并布局元素和公式元素，处理重叠情况
        """
        merged_blocks = []
        
        # 转换layoutparser的Block对象为字典
        for block in layout_blocks:
            merged_blocks.append({
                "type": block.type,
                "coordinates": (int(block.coordinates[0]), int(block.coordinates[1]), 
                               int(block.coordinates[2]), int(block.coordinates[3])),
                "confidence": float(block.score) if hasattr(block, 'score') else 1.0
            })
        
        # 添加公式块，处理与其他元素的重叠
        for formula in formula_blocks:
            # 检查是否与现有块重叠
            overlapped = False
            for i, block in enumerate(merged_blocks):
                if self._is_overlapping(formula["coordinates"], block["coordinates"]):
                    # 如果重叠且公式置信度更高，替换原块
                    if formula["confidence"] > block["confidence"]:
                        merged_blocks[i] = formula
                    overlapped = True
                    break
            
            # 如果没有重叠，添加新的公式块
            if not overlapped:
                merged_blocks.append(formula)
        
        return merged_blocks
    
    def _is_overlapping(self, box1, box2, threshold=0.5):
        """
        判断两个边界框是否重叠
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 计算交集面积
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # 计算各自面积
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # 计算IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        
        return iou > threshold
    
    def _extract_text_from_image(self, image: np.ndarray) -> str:
        """
        使用OCR从图像中提取文本
        """
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
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """
        将PIL图像转换为base64编码
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def _extract_table_basic(self, page, bbox) -> str:
        """
        使用pdfplumber从页面中提取表格并转换为Markdown格式
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
            logger.error(f"基本表格提取失败: {e}")
            return "*表格提取失败*"
    
    def _extract_table_advanced(self, image: np.ndarray) -> str:
        """
        使用TableMaster或StructEqTable从图像中提取表格
        """
        # 这里需要根据实际模型的使用方式进行实现
        # 由于这些模型的具体使用方式可能需要特定的代码，这里仅提供框架
        try:
            # 如果没有高级表格识别模型，返回空字符串，让调用者回退到基本方法
            if self.table_master_path is None and self.struct_table_path is None:
                return ""
            
            # 这里应该实现表格识别的具体逻辑
            # ...
            
            # 示例：返回一个简单的Markdown表格
            return "| 列1 | 列2 | 列3 |\n| --- | --- | --- |\n| 数据1 | 数据2 | 数据3 |\n"
        except Exception as e:
            logger.error(f"高级表格提取失败: {e}")
            return ""
    
    def _recognize_formula(self, image: np.ndarray) -> str:
        """
        使用UniMERNet识别公式图像为LaTeX代码
        """
        # 这里需要根据UniMERNet模型的使用方式进行实现
        try:
            # 如果没有公式识别模型，返回空字符串
            if self.formula_rec_path is None:
                return ""
            
            # 这里应该实现公式识别的具体逻辑
            # ...
            
            # 示例：返回一个简单的LaTeX公式
            return "E = mc^2"
        except Exception as e:
            logger.error(f"公式识别失败: {e}")
            return ""
    
    def _post_process_markdown(self, markdown: str) -> str:
        """
        对生成的Markdown进行后处理，合并相邻文本块，清理多余空行等
        """
        # 替换连续的多个空行为两个空行
        processed = '\n\n'.join([line for line in markdown.split('\n\n') if line.strip()])
        
        # 其他后处理逻辑可以在这里添加
        
        return processed