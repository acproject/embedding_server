import fitz
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class PDFPreprocessor:
    """PDF预处理器，提取PDF源数据结构信息"""
    
    def __init__(self):
        self.font_stats = {}
        self.margin_stats = {}
        self.anchors = []
    
    def analyze_pdf_structure(self, pdf_path: str) -> Dict:
        """分析PDF结构，提取字体、边距和锚点信息"""
        doc = fitz.open(pdf_path)
        structure_info = {
            "pages": [],
            "fonts": {},
            "margins": {},
            "anchors": []
        }
        
        # 遍历所有页面
        for page_idx, page in enumerate(doc):
            page_info = self._analyze_page_structure(page)
            structure_info["pages"].append(page_info)
            
            # 更新字体统计
            for font, stats in page_info["fonts"].items():
                if font not in structure_info["fonts"]:
                    structure_info["fonts"][font] = {"count": 0, "sizes": []}
                structure_info["fonts"][font]["count"] += stats["count"]
                structure_info["fonts"][font]["sizes"].extend(stats["sizes"])
        
        # 分析全局字体统计，推断标题、正文等
        structure_info["font_roles"] = self._infer_font_roles(structure_info["fonts"])
        
        # 计算全局边距
        structure_info["margins"] = self._calculate_global_margins(structure_info["pages"])
        
        doc.close()
        return structure_info
    
    def _analyze_page_structure(self, page) -> Dict:
        """分析单页结构"""
        page_info = {
            "width": page.rect.width,
            "height": page.rect.height,
            "fonts": {},
            "blocks": [],
            "margins": self._detect_margins(page),
            "anchors": []
        }
        
        # 提取文本块
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            block_info = {
                "bbox": block["bbox"],
                "type": self._guess_block_type(block),
                "lines": []
            }
            
            # 处理文本块
            if "lines" in block:
                for line in block["lines"]:
                    line_fonts = {}
                    for span in line["spans"]:
                        font = span["font"]
                        size = span["size"]
                        
                        # 更新字体统计
                        if font not in page_info["fonts"]:
                            page_info["fonts"][font] = {"count": 0, "sizes": []}
                        page_info["fonts"][font]["count"] += 1
                        page_info["fonts"][font]["sizes"].append(size)
                        
                        # 更新行字体信息
                        if font not in line_fonts:
                            line_fonts[font] = {"count": 0, "sizes": []}
                        line_fonts[font]["count"] += 1
                        line_fonts[font]["sizes"].append(size)
                    
                    line_info = {
                        "bbox": line["bbox"],
                        "text": "".join([span["text"] for span in line["spans"]]),
                        "fonts": line_fonts
                    }
                    block_info["lines"].append(line_info)
            
            page_info["blocks"].append(block_info)
            
            # 识别可能的锚点（如图表标题、章节标题等）
            if block_info["type"] in ["title", "figure_caption", "table_caption"]:
                page_info["anchors"].append({
                    "type": block_info["type"],
                    "bbox": block_info["bbox"],
                    "text": "".join([line["text"] for line in block_info["lines"]])
                })
        
        return page_info
    
    def _guess_block_type(self, block) -> str:
        """根据块特征猜测类型"""
        # 如果是图片块
        if block.get("type") == 1:
            return "image"
        
        # 文本块分析
        if "lines" in block and len(block["lines"]) > 0:
            # 提取所有span的字体和大小
            fonts = []
            sizes = []
            text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    fonts.append(span["font"])
                    sizes.append(span["size"])
                    text += span["text"]
            
            # 如果字体大小明显大于平均值，可能是标题
            if len(sizes) > 0 and np.mean(sizes) > 12:
                if text.strip().lower().startswith(("figure", "fig.", "table", "tab.")):
                    return "figure_caption" if "figure" in text.lower() or "fig." in text.lower() else "table_caption"
                return "title"
            
            # 其他文本块
            return "text"
        
        return "unknown"
    
    def _detect_margins(self, page) -> Dict:
        """检测页面边距"""
        blocks = page.get_text("dict")["blocks"]
        if not blocks:
            return {"left": 0, "right": 0, "top": 0, "bottom": 0}
        
        # 初始化边界为页面大小
        left = page.rect.width
        right = 0
        top = page.rect.height
        bottom = 0
        
        # 遍历所有块，找出文本的边界
        for block in blocks:
            if "lines" in block and len(block["lines"]) > 0:
                x0, y0, x1, y1 = block["bbox"]
                left = min(left, x0)
                right = max(right, x1)
                top = min(top, y0)
                bottom = max(bottom, y1)
        
        return {
            "left": left,
            "right": page.rect.width - right,
            "top": top,
            "bottom": page.rect.height - bottom
        }
    
    def _infer_font_roles(self, fonts) -> Dict:
        """推断字体角色（标题、正文等）"""
        font_roles = {}
        
        # 按使用频率排序字体
        sorted_fonts = sorted(fonts.items(), key=lambda x: x[1]["count"], reverse=True)
        
        # 最常用的字体可能是正文
        if sorted_fonts:
            main_font, main_stats = sorted_fonts[0]
            font_roles[main_font] = "body"
            
            # 查找比正文大的字体，可能是标题
            main_size = np.median(main_stats["sizes"])
            for font, stats in sorted_fonts[1:]:
                font_size = np.median(stats["sizes"])
                if font_size > main_size * 1.2:  # 比正文大20%以上
                    font_roles[font] = "heading"
                elif font_size < main_size * 0.9:  # 比正文小10%以上
                    font_roles[font] = "footnote"
                else:
                    font_roles[font] = "body"
        
        return font_roles
    
    def _calculate_global_margins(self, pages) -> Dict:
        """计算全局边距"""
        if not pages:
            return {"left": 0, "right": 0, "top": 0, "bottom": 0}
        
        # 收集所有页面的边距
        left_margins = [page["margins"]["left"] for page in pages]
        right_margins = [page["margins"]["right"] for page in pages]
        top_margins = [page["margins"]["top"] for page in pages]
        bottom_margins = [page["margins"]["bottom"] for page in pages]
        
        # 使用中位数作为全局边距
        return {
            "left": np.median(left_margins),
            "right": np.median(right_margins),
            "top": np.median(top_margins),
            "bottom": np.median(bottom_margins)
        }