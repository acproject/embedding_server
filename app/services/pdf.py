# 在 process_pdf_with_visualization 方法中
def process_pdf_with_visualization(self, pdf_content):
    """
    处理PDF并返回嵌入向量和可视化图像
    
    Args:
        pdf_content: PDF文件内容
        
    Returns:
        tuple: (embedding, visualization_images)
    """
    try:
        # 确保传递正确的参数给text_embedder.get_embedding方法
        # 注意：这里需要传递content_type参数
        return self.text_embedder.get_embedding(pdf_content, content_type="pdf", visualize=True)
    except Exception as e:
        logger.error(f"PDF可视化处理失败: {e}")
        raise