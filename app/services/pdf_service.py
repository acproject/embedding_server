"""
PDF处理服务模块

此模块已重构为模块化结构，请使用新的导入方式：
from app.services.pdf import PDFService
"""

from app.services.pdf.core import PDFService

# 为了向后兼容，保留原始导入方式
__all__ = ['PDFService']