import os
import logging
import numpy as np
import torch
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class LayoutAnalyzer:
    """处理PDF页面布局分析"""
    
    def __init__(self, models_dir: str, device: str = "cpu"):
        """
        初始化布局分析模块
        
        Args:
            models_dir: 模型目录
            device: 计算设备 ("cpu" 或 "cuda")
        """
        self.models_dir = models_dir
        self.device = device
        self.layout_model = None
        self.formula_extractor = None
        
        try:
            self._init_layout_models()
        except Exception as e:
            logger.error(f"布局分析模型初始化失败: {e}")
            # 如果布局分析失败，将使用基本文本提取
            self.layout_model = None
            logger.warning("将使用基本文本提取模式")
    
    def _init_layout_models(self):
        """初始化布局分析相关模型"""
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
    
    def analyze_page(self, image: np.ndarray, target_size=None) -> List[Dict]:
        """
        分析页面布局
        
        Args:
            image: 页面图像的numpy数组
            target_size: 目标图像尺寸 (width, height)，用于坐标转换
                
        Returns:
            List[Dict]: 布局元素列表，每个元素包含类型、坐标等信息
        """
        if self.layout_model is None:
            logger.warning("布局分析模型未加载，返回空布局")
            return []
        
        try:
            # 使用布局分析模型分析页面结构
            results = self.layout_model(image)
            layout = []
            
            # 获取当前图像尺寸
            img_h, img_w = image.shape[:2]
            
            # 计算坐标转换比例（如果需要）
            scale_x, scale_y = 1.0, 1.0
            if target_size is not None:
                target_w, target_h = target_size
                scale_x = target_w / img_w
                scale_y = target_h / img_h
                logger.info(f"应用坐标转换: 从 {img_w}x{img_h} 到 {target_w}x{target_h}, 比例: {scale_x:.2f}x{scale_y:.2f}")
            
            for result in results:
                for box in result.boxes:
                    # 获取坐标
                    coords = box.xyxy[0].tolist()
                    
                    # 应用坐标转换
                    x0 = int(coords[0] * scale_x)
                    y0 = int(coords[1] * scale_y)
                    x1 = int(coords[2] * scale_x)
                    y1 = int(coords[3] * scale_y)
                    
                    layout.append({
                        "type": result.names[int(box.cls)],
                        "coordinates": [x0, y0, x1, y1],
                        "confidence": float(box.conf)
                    })
            
            # 检测公式
            formula_blocks = self._detect_formulas(image)
            
            # 对公式块也应用相同的坐标转换
            if target_size is not None:
                for block in formula_blocks:
                    coords = block["coordinates"]
                    block["coordinates"] = [
                        int(coords[0] * scale_x),
                        int(coords[1] * scale_y),
                        int(coords[2] * scale_x),
                        int(coords[3] * scale_y)
                    ]
            
            # 合并布局元素和公式元素
            all_blocks = layout + formula_blocks
            
            # 按照从上到下的顺序排序布局元素
            sorted_blocks = sorted(all_blocks, key=lambda x: x["coordinates"][1])
            
            return sorted_blocks
        except Exception as e:
            logger.error(f"页面布局分析失败: {e}")
            return []
        finally:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _detect_formulas(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的公式
        
        Args:
            image: 输入图像的numpy数组
            
        Returns:
            List[Dict]: 检测到的公式列表
        """
        # 如果有公式提取器实例，则使用它
        if self.formula_extractor is not None:
            return self.formula_extractor.detect_formulas(image)
        return []
    
    def _merge_blocks(self, layout_blocks, formula_blocks) -> List[Dict]:
        """
        合并布局元素和公式元素，处理重叠情况
        
        Args:
            layout_blocks: 布局分析得到的元素列表
            formula_blocks: 公式检测得到的元素列表
            
        Returns:
            List[Dict]: 合并后的布局元素列表
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
            
            # 如果没有重叠，直接添加公式块
            if not overlapped:
                merged_blocks.append(formula)
        
        return merged_blocks
    
    def _is_overlapping(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> bool:
        """
        检查两个边界框是否重叠
        
        Args:
            box1: 第一个边界框 (x1, y1, x2, y2)
            box2: 第二个边界框 (x1, y1, x2, y2)
            
        Returns:
            bool: 如果重叠则返回True，否则返回False
        """
        # 计算交集区域
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 如果交集区域有效（宽度和高度都大于0），则存在重叠
        if x2 > x1 and y2 > y1:
            # 计算交集面积
            intersection_area = (x2 - x1) * (y2 - y1)
            
            # 计算两个框的面积
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            
            # 计算交并比 (IoU)
            iou = intersection_area / float(box1_area + box2_area - intersection_area)
            
            # 如果IoU大于阈值，则认为重叠
            return iou > 0.3
        
        return False
    
    def set_formula_extractor(self, formula_extractor):
        """
        设置公式提取器实例
        
        Args:
            formula_extractor: 公式提取器实例
        """
        self.formula_extractor = formula_extractor
    
    def analyze_layout_structure(self, blocks: List[Dict]) -> Dict:
        """
        分析布局结构，识别标题、段落、列表等
        
        Args:
            blocks: 布局元素列表
            
        Returns:
            Dict: 结构化的布局信息
        """
        structure = {
            "title": None,
            "sections": [],
            "current_section": {"heading": None, "content": []}
        }
        
        # 按照从上到下的顺序处理块
        for block in blocks:
            block_type = block.get("type", "").lower()
            
            # 处理标题
            if block_type in ["title", "heading"]:
                # 如果是文档的第一个标题，设置为文档标题
                if structure["title"] is None:
                    structure["title"] = block
                else:
                    # 如果当前部分有内容，保存它并开始新部分
                    if structure["current_section"]["content"]:
                        structure["sections"].append(structure["current_section"])
                    
                    # 创建新部分
                    structure["current_section"] = {
                        "heading": block,
                        "content": []
                    }
            # 处理其他内容
            else:
                structure["current_section"]["content"].append(block)
        
        # 添加最后一个部分
        if structure["current_section"]["content"]:
            structure["sections"].append(structure["current_section"])
        
        return structure
    
    def detect_reading_order(self, blocks: List[Dict]) -> List[Dict]:
        """
        检测块的阅读顺序
        
        Args:
            blocks: 布局元素列表
            
        Returns:
            List[Dict]: 按阅读顺序排序的布局元素列表
        """
        # 首先按照从上到下的顺序排序
        vertical_sorted = sorted(blocks, key=lambda x: x["coordinates"][1])
        
        # 识别多列布局
        columns = self._detect_columns(vertical_sorted)
        
        # 如果检测到多列，按列处理
        if len(columns) > 1:
            ordered_blocks = []
            for column in columns:
                # 对每列中的块按从上到下排序
                column_blocks = sorted(column, key=lambda x: x["coordinates"][1])
                ordered_blocks.extend(column_blocks)
            return ordered_blocks
        
        # 单列情况，直接返回垂直排序的结果
        return vertical_sorted
    
    def _detect_columns(self, blocks: List[Dict]) -> List[List[Dict]]:
        """
        检测文档中的列
        
        Args:
            blocks: 布局元素列表
            
        Returns:
            List[List[Dict]]: 按列分组的布局元素列表
        """
        if not blocks:
            return [[]]
        
        # 计算页面宽度
        page_width = max([block["coordinates"][2] for block in blocks])
        
        # 初始化列
        columns = []
        current_column = []
        
        # 计算每个块的中心x坐标
        for block in blocks:
            x1, _, x2, _ = block["coordinates"]
            center_x = (x1 + x2) / 2
            
            # 检查是否属于现有列
            column_found = False
            for i, column in enumerate(columns):
                if column:
                    # 计算列的平均中心x坐标
                    column_center = sum([(b["coordinates"][0] + b["coordinates"][2]) / 2 for b in column]) / len(column)
                    # 如果块的中心接近列的中心，将其添加到该列
                    if abs(center_x - column_center) < page_width * 0.2:  # 20%的页面宽度作为阈值
                        columns[i].append(block)
                        column_found = True
                        break
            
            # 如果不属于任何现有列，创建新列
            if not column_found:
                columns.append([block])
        
        # 按照从左到右的顺序排序列
        columns.sort(key=lambda col: sum([(b["coordinates"][0] + b["coordinates"][2]) / 2 for b in col]) / len(col) if col else 0)
        
        return columns
    
    def visualize_layout(self, image: np.ndarray, blocks: List[Dict]) -> np.ndarray:
        """
        可视化布局分析结果
        
        Args:
            image: 输入图像的numpy数组
            blocks: 布局元素列表
            
        Returns:
            np.ndarray: 可视化后的图像
        """
        try:
            import cv2
            
            # 创建图像副本
            viz_img = image.copy()
            
            # 为不同类型的块定义颜色
            colors = {
                "text": (0, 255, 0),      # 绿色
                "title": (255, 0, 0),     # 红色
                "heading": (255, 0, 0),   # 红色
                "figure": (0, 255, 255),  # 黄色
                "table": (0, 0, 255),     # 蓝色
                "formula": (255, 0, 255), # 紫色
                "list": (255, 128, 0),    # 橙色
                "header": (128, 128, 0),  # 暗黄色
                "footer": (0, 128, 128),  # 青色
            }
            
            # 绘制每个块
            for block in blocks:
                block_type = block.get("type", "").lower()
                x1, y1, x2, y2 = block["coordinates"]
                
                # 获取块的颜色，如果类型未定义则使用灰色
                color = colors.get(block_type, (128, 128, 128))
                
                # 绘制矩形
                cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
                
                # 添加标签
                label = f"{block_type} ({block.get('confidence', 0):.2f})"
                cv2.putText(viz_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            return viz_img
        except Exception as e:
            logger.error(f"布局可视化失败: {e}")
            return image


def analyze_page_with_prior(self, image: np.ndarray, page=None, pdf_structure=None, dpi=200) -> List[Dict]:
    """
    结合PDF源数据和布局模型分析页面
    
    Args:
        image: 页面图像的numpy数组
        page: PyMuPDF页面对象
        pdf_structure: PDF预处理得到的结构信息
        dpi: 图像的DPI值
            
    Returns:
        List[Dict]: 布局元素列表
    """
    # 获取模型预测的布局
    model_layout = self.analyze_page(image, page, dpi)
    
    # 如果没有PDF源数据，直接返回模型结果
    if page is None or pdf_structure is None:
        return model_layout
    
    # 获取当前页面的结构信息
    page_idx = page.number
    if page_idx >= len(pdf_structure["pages"]):
        return model_layout
    
    page_structure = pdf_structure["pages"][page_idx]
    
    # 计算PDF坐标到像素坐标的转换比例
    scale_factor = dpi / 72.0
    img_h, img_w = image.shape[:2]
    
    # 从PDF源数据中提取布局元素
    pdf_layout = []
    for block in page_structure["blocks"]:
        # 转换坐标
        x0, y0, x1, y1 = block["bbox"]
        x0 = int(x0 * scale_factor)
        y0 = int(y0 * scale_factor)
        x1 = int(x1 * scale_factor)
        y1 = int(y1 * scale_factor)
        
        # 确保坐标在图像范围内
        x0 = max(0, min(x0, img_w-1))
        y0 = max(0, min(y0, img_h-1))
        x1 = max(0, min(x1, img_w-1))
        y1 = max(0, min(y1, img_h-1))
        
        pdf_layout.append({
            "type": block["type"],
            "coordinates": [x0, y0, x1, y1],
            "confidence": 1.0,  # PDF源数据的置信度设为1.0
            "source": "pdf"  # 标记来源
        })
    
    # 为模型结果添加来源标记
    for item in model_layout:
        item["source"] = "model"
    
    # 整合两种结果
    merged_layout = self._merge_layouts(model_layout, pdf_layout)
    
    return merged_layout

def _merge_layouts(self, model_layout, pdf_layout) -> List[Dict]:
    """
    整合模型预测和PDF源数据的布局结果
    
    策略：
    1. 对于重叠度高的元素，优先选择置信度高的
    2. 对于PDF源数据中的特殊元素（如图表标题），保留下来
    3. 对于模型检测到但PDF源数据中没有的元素，根据置信度决定是否保留
    """
    merged = []
    used_pdf_indices = set()
    
    # 首先处理模型结果
    for model_item in model_layout:
        model_box = model_item["coordinates"]
        best_match = None
        best_iou = 0
        best_idx = -1
        
        # 寻找与模型结果最匹配的PDF元素
        for i, pdf_item in enumerate(pdf_layout):
            if i in used_pdf_indices:
                continue
                
            pdf_box = pdf_item["coordinates"]
            iou = self._calculate_iou(model_box, pdf_box)
            
            if iou > 0.5 and iou > best_iou:  # IOU阈值设为0.5
                best_match = pdf_item
                best_iou = iou
                best_idx = i
        
        if best_match:
            # 如果找到匹配，选择置信度高的或特定类型
            used_pdf_indices.add(best_idx)
            
            # 特殊处理某些类型
            if best_match["type"] in ["figure_caption", "table_caption"]:
                merged.append(best_match)
            elif model_item["confidence"] >= 0.8:  # 模型高置信度
                merged.append(model_item)
            else:
                merged.append(best_match)
        else:
            # 没有匹配，保留模型结果
            if model_item["confidence"] >= 0.5:  # 置信度阈值
                merged.append(model_item)
    
    # 添加未使用的PDF元素
    for i, pdf_item in enumerate(pdf_layout):
        if i not in used_pdf_indices:
            merged.append(pdf_item)
    
    # 按照从上到下的顺序排序
    sorted_merged = sorted(merged, key=lambda x: x["coordinates"][1])
    
    return sorted_merged

def _calculate_iou(self, box1, box2) -> float:
    """计算两个边界框的IOU"""
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    
    # 计算交集区域
    x0_i = max(x0_1, x0_2)
    y0_i = max(y0_1, y0_2)
    x1_i = min(x1_1, x1_2)
    y1_i = min(y1_1, y1_2)
    
    # 如果没有交集
    if x0_i >= x1_i or y0_i >= y1_i:
        return 0.0
    
    # 计算交集面积
    area_i = (x1_i - x0_i) * (y1_i - y0_i)
    
    # 计算两个框的面积
    area_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    
    # 计算并集面积
    area_u = area_1 + area_2 - area_i
    
    # 返回IOU
    return area_i / area_u if area_u > 0 else 0.0
