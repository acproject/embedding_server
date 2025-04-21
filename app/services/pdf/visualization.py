import io
import base64
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class PDFVisualization:
    """处理PDF可视化相关功能"""
    
    def __init__(self):
        """初始化可视化模块"""
        # 尝试加载字体，如果失败则使用默认字体
        try:
            self.font = ImageFont.truetype("Arial", 24)
            self.small_font = ImageFont.truetype("Arial", 16)
        except IOError:
            self.font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()
        
        # 为不同类型的布局元素定义颜色
        self.type_colors = {
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
            "footer": (100, 149, 237),      # 淡蓝色
            "other": (128, 128, 128)        # 灰色
        }
        
        # 为不同类型的块定义颜色
        self.block_colors = {
            0: (255, 0, 0, 64),    # 文本块 - 红色
            1: (0, 255, 0, 64),    # 图像块 - 绿色
            2: (0, 0, 255, 64),    # 表格块 - 蓝色
            3: (255, 255, 0, 64),  # 公式块 - 黄色
        }
        
        # 块类型映射
        self.block_types = {
            0: "文本", 
            1: "图像", 
            2: "表格", 
            3: "公式"
        }
        
        # 来源样式
        self.source_styles = {
            "model": {"width": 3, "dash": None},
            "pdf": {"width": 2, "dash": (5, 5)}  # 虚线
        }
    
    def bytes_to_pil_image(self, img_data: bytes) -> Image.Image:
        """将字节数据转换为PIL图像"""
        return Image.open(io.BytesIO(img_data))
    
    def resize_image_to_fit(self, image: Image.Image, max_width: int = 800, max_height: int = 1024) -> Image.Image:
        """
        调整图像大小，使其适合指定的最大尺寸，同时保持纵横比
        
        Args:
            image: PIL图像对象
            max_width: 最大宽度
            max_height: 最大高度
            
        Returns:
            PIL图像对象: 调整大小后的图像
        """
        # 获取原始尺寸
        width, height = image.size
        
        # 如果图像已经在限制范围内，则不需要调整
        if width <= max_width and height <= max_height:
            return image
        
        # 计算调整比例
        width_ratio = max_width / width
        height_ratio = max_height / height
        ratio = min(width_ratio, height_ratio)
        
        # 计算新尺寸
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # 调整图像大小
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        return resized_image
    
    def encode_image_to_base64(self, image: Image.Image, title: Optional[str] = None) -> str:
        """将PIL图像编码为base64字符串"""
        # 调整图像大小
        image = self.resize_image_to_fit(image)
        
        # 如果提供了标题，在图像上添加标题
        if title:
            draw = ImageDraw.Draw(image)
            
            # 在图像顶部添加标题
            draw.rectangle((0, 0, image.width, 40), fill=(0, 0, 0))
            draw.text((10, 10), title, fill=(255, 255, 255), font=self.font)
        
        # 将图像编码为base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    
    def image_to_base64(self, image: Image.Image) -> str:
        """将PIL图像转换为base64编码（不添加标题）"""
        # 调整图像大小
        image = self.resize_image_to_fit(image)
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def display_image(self, image: Union[Image.Image, np.ndarray], title: Optional[str] = None, figsize: Tuple[int, int] = (10, 14)) -> None:
        """
        显示图像
        
        Args:
            image: PIL图像或numpy数组
            title: 图像标题
            figsize: 图像大小
        """
        plt.figure(figsize=figsize)
        if isinstance(image, np.ndarray):
            plt.imshow(image)
        else:
            plt.imshow(np.array(image))
        
        if title:
            plt.title(title)
        
        plt.axis('off')
        plt.show()
    
    def visualize_page_layout(self, image: Image.Image, page) -> Image.Image:
        """可视化页面布局"""
        # 创建图像副本以进行绘制
        viz_img = image.copy()
        draw = ImageDraw.Draw(viz_img)
        
        # 绘制页面边界
        draw.rectangle((0, 0, viz_img.width-1, viz_img.height-1), outline=(255, 0, 0), width=2)
        
        # 获取页面布局信息
        blocks = page.get_text("blocks")
        
        # 绘制每个块
        for block in blocks:
            block_type = block[6]  # 块类型
            x0, y0, x1, y1 = block[:4]  # 块坐标
            
            # 获取块颜色
            color = self.block_colors.get(block_type, (128, 128, 128, 64))
            
            # 绘制块边界和填充
            draw.rectangle((x0, y0, x1, y1), outline=color[:3], width=2)
            draw.rectangle((x0, y0, x1, y1), fill=color)
            
            # 添加块类型标签
            label = self.block_types.get(block_type, f"类型{block_type}")
            draw.text((x0, y0-15), label, fill=(0, 0, 0), font=self.small_font)
        
        return viz_img
    
    def visualize_text_blocks(self, image: Image.Image, page) -> Image.Image:
        """可视化文本块"""
        viz_img = image.copy()
        draw = ImageDraw.Draw(viz_img)
        
        # 获取文本块
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0]  # 类型0是文本块
        
        # 绘制每个文本块
        for block in text_blocks:
            x0, y0, x1, y1 = block[:4]
            text = block[4]
            
            # 绘制文本块边界
            draw.rectangle((x0, y0, x1, y1), outline=(0, 0, 255), width=2)
            
            # 显示部分文本
            display_text = text[:20] + "..." if len(text) > 20 else text
            draw.text((x0, y0-15), display_text, fill=(0, 0, 0), font=self.small_font)
        
        return viz_img
    
    def visualize_images(self, image: Image.Image, page) -> Image.Image:
        """可视化图像区域"""
        viz_img = image.copy()
        draw = ImageDraw.Draw(viz_img)
        
        # 获取图像块
        blocks = page.get_text("blocks")
        image_blocks = [b for b in blocks if b[6] == 1]  # 类型1是图像块
        
        # 绘制每个图像块
        for i, block in enumerate(image_blocks):
            x0, y0, x1, y1 = block[:4]
            
            # 绘制图像块边界
            draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0), width=2)
            draw.text((x0, y0-15), f"图像 {i+1}", fill=(0, 0, 0), font=self.small_font)
        
        return viz_img
    
    def visualize_tables(self, image: Image.Image, page) -> Image.Image:
        """可视化表格"""
        viz_img = image.copy()
        draw = ImageDraw.Draw(viz_img)
        
        # 获取表格块
        blocks = page.get_text("blocks")
        table_blocks = [b for b in blocks if b[6] == 2]  # 类型2是表格块
        
        # 绘制每个表格块
        for i, block in enumerate(table_blocks):
            x0, y0, x1, y1 = block[:4]
            
            # 绘制表格块边界
            draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 255), width=2)
            draw.text((x0, y0-15), f"表格 {i+1}", fill=(0, 0, 0), font=self.small_font)
        
        return viz_img
    
    def visualize_formulas(self, image: Image.Image, page) -> Image.Image:
        """可视化公式"""
        viz_img = image.copy()
        draw = ImageDraw.Draw(viz_img)
        
        # 获取公式块
        blocks = page.get_text("blocks")
        formula_blocks = [b for b in blocks if b[6] == 3]  # 类型3是公式块
        
        # 绘制每个公式块
        for i, block in enumerate(formula_blocks):
            x0, y0, x1, y1 = block[:4]
            formula_text = block[4]
            
            # 绘制公式块边界
            draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 0), width=2)
            
            # 显示公式文本
            display_text = formula_text[:30] + "..." if len(formula_text) > 30 else formula_text
            draw.text((x0, y0-15), f"公式 {i+1}: {display_text}", fill=(0, 0, 0), font=self.small_font)
        
        return viz_img
    
    def visualize_merged_layout(self, image: Image.Image, layout_results: List[Dict[str, Any]], dpi: int = 200) -> Image.Image:
        """
        可视化混合布局分析结果
        
        Args:
            image: PIL图像对象
            layout_results: 混合布局分析结果列表
            dpi: 图像的DPI值
        
        Returns:
            PIL图像对象: 添加了布局标注的图像
        """
        # 创建图像副本以进行绘制
        viz_img = image.copy()
        draw = ImageDraw.Draw(viz_img)
        
        # 获取图像尺寸
        img_w, img_h = viz_img.size
        
        # 绘制每个布局元素
        for element in layout_results:
            element_type = element.get("type", "other")
            source = element.get("source", "model")
            bbox = element.get("bbox") or element.get("coordinates")
            
            if bbox:
                # 确保坐标为整数
                x0 = max(0, int(bbox[0]))
                y0 = max(0, int(bbox[1]))
                x1 = min(img_w-1, int(bbox[2]))
                y1 = min(img_h-1, int(bbox[3]))
                
                # 获取元素颜色和样式
                color = self.type_colors.get(element_type, self.type_colors["other"])
                style = self.source_styles.get(source, self.source_styles["model"])
                
                # 绘制边框
                if style["dash"]:
                    # PIL不直接支持虚线，需要手动实现
                    self._draw_dashed_rectangle(draw, [x0, y0, x1, y1], color, style["width"], style["dash"])
                else:
                    draw.rectangle([x0, y0, x1, y1], outline=color, width=style["width"])
                
                # 添加标签
                confidence = element.get("confidence", 1.0)
                label = f"{element_type} ({source}, {confidence:.2f})"
                label_y = max(0, y0 - 15)
                draw.text((x0, label_y), label, fill=color, font=self.small_font)
        
        return viz_img
    
    def _draw_dashed_rectangle(self, draw, bbox, color, width, dash):
        """绘制虚线矩形"""
        x0, y0, x1, y1 = bbox
        dash_on, dash_off = dash
        
        # 绘制上边
        x, y = x0, y0
        while x < x1:
            x_end = min(x + dash_on, x1)
            draw.line([(x, y), (x_end, y)], fill=color, width=width)
            x = x_end + dash_off
        
        # 绘制右边
        x, y = x1, y0
        while y < y1:
            y_end = min(y + dash_on, y1)
            draw.line([(x, y), (x, y_end)], fill=color, width=width)
            y = y_end + dash_off
        
        # 绘制下边
        x, y = x1, y1
        while x > x0:
            x_start = max(x - dash_on, x0)
            draw.line([(x, y), (x_start, y)], fill=color, width=width)
            x = x_start - dash_off
        
        # 绘制左边
        x, y = x0, y1
        while y > y0:
            y_start = max(y - dash_on, y0)
            draw.line([(x, y), (x, y_start)], fill=color, width=width)
            y = y_start - dash_off
    
    def visualize_layout(self, image: Image.Image, layout_results: List[Dict[str, Any]]) -> Image.Image:
        """
        可视化页面布局分析结果
        
        Args:
            image: PIL图像对象
            layout_results: 布局分析结果列表
        
        Returns:
            PIL图像对象: 添加了布局标注的图像
        """
        # 创建图像副本以进行绘制
        viz_img = image.copy()
        draw = ImageDraw.Draw(viz_img)
        
        # 获取图像尺寸
        img_w, img_h = viz_img.size
        
        # 绘制每个布局元素
        for element in layout_results:
            element_type = element.get("type", "other")
            bbox = element.get("bbox") or element.get("coordinates")
            
            if bbox:
                # 确保坐标为整数且在图像范围内
                x0 = max(0, min(int(bbox[0]), img_w-1))
                y0 = max(0, min(int(bbox[1]), img_h-1))
                x1 = max(0, min(int(bbox[2]), img_w-1))
                y1 = max(0, min(int(bbox[3]), img_h-1))
                
                # 获取元素颜色
                color = self.type_colors.get(element_type, self.type_colors["other"])
                
                # 只绘制边框，不填充
                draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                
                # 添加标签
                label_y = max(0, y0 - 15)
                draw.text((x0, label_y), element_type, fill=color, font=self.small_font)
        
        return viz_img
    
    def visualize_formula_detection(self, image: Image.Image, formula_results: Dict[str, Any]) -> Image.Image:
        """
        可视化公式检测结果
        
        Args:
            image: PIL图像对象
            formula_results: 公式检测结果
            
        Returns:
            PIL图像对象: 添加了公式检测标注的图像
        """
        viz_img = image.copy()
        draw = ImageDraw.Draw(viz_img)
        
        # 获取检测到的公式
        boxes = formula_results.get("detection", {}).get("boxes", [])
        formulas = formula_results.get("recognition", {}).get("formulas", [])
        
        # 绘制每个检测到的公式
        for i, box_info in enumerate(boxes):
            bbox = box_info.get("bbox")
            if bbox:
                x0, y0, x1, y1 = [int(coord) for coord in bbox]
                
                # 绘制边框
                draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
                
                # 添加标签
                draw.text((x0, y0-15), f"公式 {i+1}", fill=(255, 0, 0), font=self.small_font)
        
        # 如果有识别结果，显示在图像下方
        if formulas:
            # 创建一个更大的图像，底部留出空间显示识别结果
            result_height = 30 * len(formulas)
            new_height = viz_img.height + result_height + 20
            result_img = Image.new("RGB", (viz_img.width, new_height), (255, 255, 255))
            result_img.paste(viz_img, (0, 0))
            
            # 在底部显示识别结果
            draw = ImageDraw.Draw(result_img)
            draw.rectangle((0, viz_img.height, viz_img.width, new_height), fill=(240, 240, 240))
            
            y_pos = viz_img.height + 10
            for i, formula in enumerate(formulas):
                text = formula.get("text", "")
                confidence = formula.get("confidence", 0.0)
                draw.text((10, y_pos), f"公式 {i+1}: {text} (置信度: {confidence:.2f})", 
                          fill=(0, 0, 0), font=self.small_font)
                y_pos += 30
            
            return result_img
        
        return viz_img
    
    def visualize_table_detection(self, image: Image.Image, table_results: List[Dict[str, Any]]) -> Image.Image:
        """
        可视化表格检测结果
        
        Args:
            image: PIL图像对象
            table_results: 表格检测结果列表
            
        Returns:
            PIL图像对象: 添加了表格检测标注的图像
        """
        viz_img = image.copy()
        draw = ImageDraw.Draw(viz_img)
        
        # 绘制每个检测到的表格
        for i, table in enumerate(table_results):
            bbox = table.get("bbox") or table.get("coordinates")
            if bbox:
                x0, y0, x1, y1 = [int(coord) for coord in bbox]
                
                # 绘制边框
                draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 255), width=2)
                
                # 添加标签
                confidence = table.get("confidence", 1.0)
                draw.text((x0, y0-15), f"表格 {i+1} ({confidence:.2f})", fill=(0, 0, 255), font=self.small_font)
                
                # 如果有表格结构信息，绘制单元格
                cells = table.get("cells", [])
                for cell in cells:
                    cell_bbox = cell.get("bbox")
                    if cell_bbox:
                        cx0, cy0, cx1, cy1 = [int(coord) for coord in cell_bbox]
                        draw.rectangle([cx0, cy0, cx1, cy1], outline=(0, 255, 255), width=1)
        
        return viz_img
    
    def visualize_pdf_layout_info(self, image: Image.Image, layout_info, page_idx: int) -> Image.Image:
        """
        可视化PDFLayoutInfo中的布局信息
        
        Args:
            image: PIL图像对象
            layout_info: PDFLayoutInfo对象
            page_idx: 页面索引
            
        Returns:
            PIL图像对象: 添加了布局标注的图像
        """
        # 获取页面布局元素
        page_layout = layout_info.get_page_layout(page_idx)
        
        # 使用visualize_layout方法绘制布局
        viz_img = self.visualize_layout(image, page_layout)
        
        # 获取特殊元素
        formulas = layout_info.get_formulas(page_idx)
        tables = layout_info.get_tables(page_idx)
        figures = layout_info.get_figures(page_idx)
        
        # 绘制公式
        draw = ImageDraw.Draw(viz_img)
        for i, formula in enumerate(formulas):
            bbox = formula.get("bbox") or formula.get("coordinates")
            if bbox:
                x0, y0, x1, y1 = [int(coord) for coord in bbox]
                draw.rectangle([x0, y0, x1, y1], outline=(128, 0, 128), width=2)
                draw.text((x0, y0-15), f"公式 {i+1}", fill=(128, 0, 128), font=self.small_font)
        
        # 绘制表格
        for i, table in enumerate(tables):
            bbox = table.get("bbox") or table.get("coordinates")
            if bbox:
                x0, y0, x1, y1 = [int(coord) for coord in bbox]
                draw.rectangle([x0, y0, x1, y1], outline=(255, 165, 0), width=2)
                draw.text((x0, y0-15), f"表格 {i+1}", fill=(255, 165, 0), font=self.small_font)
        
        # 绘制图片
        for i, figure in enumerate(figures):
            bbox = figure.get("bbox") or figure.get("coordinates")
            if bbox:
                x0, y0, x1, y1 = [int(coord) for coord in bbox]
                draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 255), width=2)
                draw.text((x0, y0-15), f"图片 {i+1}", fill=(0, 0, 255), font=self.small_font)
        
        return viz_img
    
    def create_comparison_image(self, images: List[Image.Image], titles: List[str] = None) -> Image.Image:
        """
        创建比较图像，将多个图像并排显示
        
        Args:
            images: 图像列表
            titles: 标题列表
            
        Returns:
            PIL图像对象: 合并后的图像
        """
        if not images:
            return None
        
        # 确定图像尺寸
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)
        
        # 创建新图像
        n_images = len(images)
        result_width = max_width * n_images
        result_height = max_height + 40 if titles else max_height
        
        result = Image.new("RGB", (result_width, result_height), (255, 255, 255))
        draw = ImageDraw.Draw(result)
        
        # 添加图像和标题
        for i, img in enumerate(images):
            # 粘贴图像
            x_offset = i * max_width
            result.paste(img, (x_offset, 40 if titles else 0))
            
            # 添加标题
            if titles and i < len(titles):
                title_x = x_offset + (max_width - len(titles[i]) * 10) // 2
                draw.text((title_x, 10), titles[i], fill=(0, 0, 0), font=self.font)
                
                # 添加分隔线
                if i > 0:
                    draw.line([(x_offset, 0), (x_offset, result_height)], fill=(200, 200, 200), width=2)
        
        return result
    
    def highlight_element(self, image: Image.Image, bbox: List[float], color: Tuple[int, int, int] = (255, 0, 0), 
                         label: Optional[str] = None) -> Image.Image:
        """
        在图像中高亮显示指定元素
        
        Args:
            image: PIL图像对象
            bbox: 边界框坐标 [x0, y0, x1, y1]
            color: 高亮颜色
            label: 标签文本
            
        Returns:
            PIL图像对象: 添加了高亮的图像
        """
        viz_img = image.copy()
        draw = ImageDraw.Draw(viz_img)
        
        # 确保坐标为整数
        x0, y0, x1, y1 = [int(coord) for coord in bbox]
        
        # 绘制边框
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        
        # 添加标签
        if label:
            label_y = max(0, y0 - 20)
            draw.text((x0, label_y), label, fill=color, font=self.small_font)
        
        return viz_img
    
    def crop_element(self, image: Image.Image, bbox: List[float]) -> Image.Image:
        """
        从图像中裁剪出指定元素
        
        Args:
            image: PIL图像对象
            bbox: 边界框坐标 [x0, y0, x1, y1]
            
        Returns:
            PIL图像对象: 裁剪后的图像
        """
        # 确保坐标为整数
        x0, y0, x1, y1 = [int(coord) for coord in bbox]
        
        # 确保坐标在图像范围内
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(image.width, x1)
        y1 = min(image.height, y1)
        
        # 裁剪图像
        return image.crop((x0, y0, x1, y1))