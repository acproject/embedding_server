import io
import base64
import logging
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

class PDFVisualization:
    """处理PDF可视化相关功能"""
    
    def __init__(self):
        """初始化可视化模块"""
        pass
    
    def bytes_to_pil_image(self, img_data):
        """将字节数据转换为PIL图像"""
        return Image.open(io.BytesIO(img_data))
    
    def encode_image_to_base64(self, image, title=None):
        """将PIL图像编码为base64字符串"""
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
    
    def image_to_base64(self, image):
        """将PIL图像转换为base64编码（不添加标题）"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def visualize_page_layout(self, image, page):
        """可视化页面布局"""
        # 创建图像副本以进行绘制
        viz_img = image.copy()
        draw = ImageDraw.Draw(viz_img)
        
        # 绘制页面边界
        draw.rectangle((0, 0, viz_img.width-1, viz_img.height-1), outline=(255, 0, 0), width=2)
        
        # 获取页面布局信息
        blocks = page.get_text("blocks")
        
        # 为不同类型的块使用不同颜色
        colors = {
            0: (255, 0, 0, 64),    # 文本块 - 红色
            1: (0, 255, 0, 64),    # 图像块 - 绿色
            2: (0, 0, 255, 64),    # 表格块 - 蓝色
            3: (255, 255, 0, 64),  # 公式块 - 黄色
        }
        
        # 绘制每个块
        for block in blocks:
            block_type = block[6]  # 块类型
            x0, y0, x1, y1 = block[:4]  # 块坐标
            
            # 获取块颜色
            color = colors.get(block_type, (128, 128, 128, 64))
            
            # 绘制块边界和填充
            draw.rectangle((x0, y0, x1, y1), outline=color[:3], width=2)
            draw.rectangle((x0, y0, x1, y1), fill=color)
            
            # 添加块类型标签
            block_types = {0: "文本", 1: "图像", 2: "表格", 3: "公式"}
            label = block_types.get(block_type, f"类型{block_type}")
            draw.text((x0, y0-15), label, fill=(0, 0, 0))
        
        return viz_img
    
    def visualize_text_blocks(self, image, page):
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
            draw.text((x0, y0-15), display_text, fill=(0, 0, 0))
        
        return viz_img
    
    def visualize_images(self, image, page):
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
            draw.text((x0, y0-15), f"图像 {i+1}", fill=(0, 0, 0))
        
        return viz_img
    
    def visualize_tables(self, image, page):
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
            draw.text((x0, y0-15), f"表格 {i+1}", fill=(0, 0, 0))
        
        return viz_img
    
    def visualize_formulas(self, image, page):
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
            draw.text((x0, y0-15), f"公式 {i+1}: {display_text}", fill=(0, 0, 0))
        
        return viz_img
    
    def visualize_layout(self, image, layout_results):
        """可视化页面布局分析结果"""
        # 创建图像副本以进行绘制
        viz_img = image.copy()
        draw = ImageDraw.Draw(viz_img)
        
        # 为不同类型的布局元素使用不同颜色
        colors = {
            "title": (255, 0, 0, 128),      # 红色
            "text": (0, 255, 0, 128),       # 绿色
            "table": (0, 0, 255, 128),      # 蓝色
            "figure": (255, 255, 0, 128),   # 黄色
            "formula": (255, 0, 255, 128),  # 紫色
            "header": (0, 255, 255, 128),   # 青色
            "footer": (255, 128, 0, 128),   # 橙色
            "other": (128, 128, 128, 128)   # 灰色
        }
        
        # 绘制每个布局元素
        for element in layout_results:
            element_type = element.get("type", "other")
            bbox = element.get("bbox")
            
            if bbox:
                color = colors.get(element_type, colors["other"])
                draw.rectangle(bbox, outline=color[:3], width=3)
                draw.rectangle(bbox, fill=color)
                
                # 添加标签
                label = element_type
                draw.text((bbox[0], bbox[1] - 15), label, fill=(0, 0, 0))
        
        return viz_img