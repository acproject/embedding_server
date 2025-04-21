import os
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import torch
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

class LayoutAnalyzer:
    """处理PDF布局分析"""
    
    def __init__(self, models_dir: str, device: str = "cpu"):
        """
        初始化布局分析模块
        
        Args:
            models_dir: 模型目录
            device: 计算设备 ("cpu" 或 "cuda")
        """
        self.models_dir = models_dir
        self.device = device
        self.model = None
        self.model_input_size = (800, 1024)  # 默认模型输入尺寸
        
        try:
            self._init_layout_model()
        except Exception as e:
            logger.error(f"布局分析模型初始化失败: {e}")
            raise
    
    def _init_layout_model(self):
        """初始化布局分析模型"""
        try:
            # 尝试导入YOLO
            import ultralytics
            from ultralytics import YOLO
            
            # 布局分析模型路径
            layout_model_path = os.path.join(self.models_dir, "layout/DocLayout-YOLO_ft/yolov10l_ft.pt")
            
            # 检查模型文件是否存在
            if not os.path.exists(layout_model_path):
                logger.warning(f"布局模型文件不存在: {layout_model_path}")
                # 尝试使用默认YOLO模型
                self.model = YOLO("yolov8n.pt")
                logger.info("使用默认YOLO模型")
            else:
                # 加载自定义布局模型
                self.model = YOLO(layout_model_path)
                logger.info(f"布局模型加载成功: {layout_model_path}")
            
            # 设置设备
            self.model.to(self.device)
            
            # 获取模型输入尺寸
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'args'):
                if hasattr(self.model.model.args, 'imgsz'):
                    imgsz = self.model.model.args.imgsz
                    if isinstance(imgsz, (list, tuple)) and len(imgsz) >= 2:
                        self.model_input_size = (imgsz[1], imgsz[0])  # 宽度, 高度
                    else:
                        self.model_input_size = (imgsz, imgsz)  # 正方形
            
            logger.info(f"布局模型输入尺寸: {self.model_input_size}")
            
        except ImportError:
            logger.error("未安装ultralytics，无法使用YOLO进行布局分析")
            raise
    
    def analyze_page(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        分析页面布局
        
        Args:
            image: 页面图像，numpy数组格式
            
        Returns:
            List[Dict]: 布局元素列表，每个元素包含类型、坐标和置信度
        """
        if self.model is None:
            logger.error("布局分析模型未初始化")
            return []
        
        try:
            # 确保图像是RGB格式
            if len(image.shape) == 2:
                # 灰度图转RGB
                image = np.stack([image] * 3, axis=2)
            elif image.shape[2] == 4:
                # RGBA转RGB
                image = image[:, :, :3]
            
            # 使用YOLO模型进行预测
            results = self.model(image, verbose=False)
            
            # 解析结果
            layout_elements = []
            
            for result in results:
                boxes = result.boxes
                for i in range(len(boxes)):
                    # 获取边界框坐标 (x1, y1, x2, y2)
                    box = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = box
                    
                    # 获取类别和置信度
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    
                    # 获取类别名称
                    cls_name = result.names[cls_id]
                    
                    # 添加到布局元素列表
                    layout_elements.append({
                        "type": cls_name,
                        "coordinates": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": conf
                    })
            
            return layout_elements
            
        except Exception as e:
            logger.error(f"布局分析失败: {e}")
            return []
    
    def visualize_layout(self, image: np.ndarray, layout_elements: List[Dict[str, Any]]) -> Image.Image:
        """
        可视化布局分析结果
        
        Args:
            image: 页面图像，numpy数组格式
            layout_elements: 布局元素列表
            
        Returns:
            Image.Image: 带有布局标注的图像
        """
        # 转换为PIL图像
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype('uint8'))
        else:
            img = image
        
        draw = ImageDraw.Draw(img)
        
        # 为不同类型的元素使用不同颜色
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
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # 绘制每个布局元素
        for i, element in enumerate(layout_elements):
            element_type = element["type"].lower()
            coords = element["coordinates"]
            confidence = element.get("confidence", 0)
            
            # 获取颜色
            color = colors.get(element_type, (200, 200, 200))
            
            # 绘制边界框
            draw.rectangle(coords, outline=color, width=2)
            
            # 添加标签
            label = f"{i+1}.{element_type} ({confidence:.2f})"
            draw.text((coords[0], coords[1]-20), label, fill=color, font=font)
        
        return img


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
