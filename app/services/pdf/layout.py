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
    
    def analyze_page(self, image: np.ndarray) -> List[Dict]:
        """
        分析页面布局
        
        Args:
            image: 页面图像的numpy数组
            
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
            
            for result in results:
                for box in result.boxes:
                    layout.append({
                        "type": result.names[int(box.cls)],
                        "coordinates": [int(x) for x in box.xyxy[0].tolist()],
                        "confidence": float(box.conf)
                    })
            
            # 检测公式
            formula_blocks = self._detect_formulas(image)
            
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