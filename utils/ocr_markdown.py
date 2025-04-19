from PIL import Image, ImageDraw, ImageFont  # 添加ImageDraw和ImageFont的导入
import fitz  # PyMuPDF
import io
import base64
import logging
def extract_image_as_base64(page, bbox):
    """
    从PDF页面提取图片并转换为base64编码
    
    Args:
        page: PDF页面对象
        bbox: 图片边界框
        
    Returns:
        包含图片信息的字典，包括base64编码
    """
    try:
        # 从页面提取图片区域
        rect = fitz.Rect(bbox)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
        
        # 转换为PIL图像
        img_data = pix.samples
        img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
        
        # 转换为base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "bbox": bbox,
            "base64": img_base64,
            "width": pix.width,
            "height": pix.height,
            "format": "PNG"
        }
    except Exception as e:
        logging.error(f"图片转base64失败: {str(e)}")
        return None

# 为TextExtractor类添加改进文本空格处理的方法
def improve_text_spacing(self, text):
    """
    改进OCR文本的空格处理
    
    Args:
        text: 原始OCR文本
        
    Returns:
        改进后的文本
    """
    if not text:
        return text
    
    # 1. 在中英文之间添加空格
    import re
    text = re.sub(r'([a-zA-Z0-9])([^\sa-zA-Z0-9])', r'\1 \2', text)
    text = re.sub(r'([^\sa-zA-Z0-9])([a-zA-Z0-9])', r'\1 \2', text)
    
    # 2. 修复多余的空格
    text = re.sub(r'\s+', ' ', text)
    
    # 3. 修复标点符号前的空格
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    
    # 4. 确保句子之间有适当的空格
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    return text.strip()
