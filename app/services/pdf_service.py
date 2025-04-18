"""
PDF处理服务模块

此模块已重构为模块化结构，请使用新的导入方式：
from app.services.pdf import PDFService
"""

from app.services.pdf.core import PDFService

def convert_pdf_to_markdown(pdf_content: bytes) -> str:
    """
    将PDF文档转换为Markdown格式（统一入口）
    Args:
        pdf_content: PDF文档的二进制内容
    Returns:
        str: 转换后的Markdown文本
    """
    service = PDFService(models_dir="models", text_embedder=None)  # 请根据实际情况传递参数
    return service.convert_pdf_to_markdown(pdf_content)

# 为了向后兼容，保留原始导入方式
__all__ = ['PDFService']