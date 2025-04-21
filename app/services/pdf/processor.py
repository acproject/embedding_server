# 假设这是你的PDF处理主函数
def process_pdf(pdf_bytes, output_path):
    # 加载PDF
    images = load_images_from_pdf(pdf_bytes)
    
    for i, img_dict in enumerate(images):
        # 获取原始图像
        image = img_dict['img']
        original_width = img_dict['width']
        original_height = img_dict['height']
        
        # 记录最终输出尺寸
        final_width, final_height = 800, 1024  # 你提到的最终尺寸
        
        # 分析布局时传入目标尺寸
        layout_results = layout_analyzer.analyze_page(
            image, 
            target_size=(final_width, final_height)
        )
        
        # 可视化布局
        # 注意：如果可视化前图像已经被调整为final_width x final_height，
        # 则不需要再次转换坐标
        pil_image = Image.fromarray(image)
        if pil_image.size != (final_width, final_height):
            pil_image = pil_image.resize((final_width, final_height))
            
        visualized_image = pdf_visualizer.visualize_layout(pil_image, layout_results)
        
        # 保存结果
        visualized_image.save(f"{output_path}/page_{i}.png")


def process_pdf_page(page_image, layout_analyzer, pdf_visualizer, final_size=(1191, 1582)):
    """
    处理单个PDF页面
    
    Args:
        page_image: 页面图像的numpy数组
        layout_analyzer: 布局分析器实例
        pdf_visualizer: PDF可视化器实例
        final_size: 最终输出图像的尺寸 (width, height)
    
    Returns:
        PIL图像对象: 添加了布局标注的图像
    """
    # 获取原始图像尺寸
    img_h, img_w = page_image.shape[:2]
    
    # 分析布局时传入目标尺寸
    layout_results = layout_analyzer.analyze_page(
        page_image, 
        target_size=final_size
    )
    
    # 将numpy数组转换为PIL图像
    pil_image = Image.fromarray(page_image)
    
    # 如果需要，调整图像尺寸
    if (pil_image.width, pil_image.height) != final_size:
        pil_image = pil_image.resize(final_size)
    
    # 可视化布局
    visualized_image = pdf_visualizer.visualize_layout(pil_image, layout_results)
    
    return visualized_image