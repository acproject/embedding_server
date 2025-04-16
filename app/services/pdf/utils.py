def pdf_coord_to_pixel(pdf_coord, pdf_size, img_size):
    """
    将PDF坐标转换为像素坐标
    
    Args:
        pdf_coord: PDF坐标 [x0, y0, x1, y1]
        pdf_size: PDF页面尺寸 (width, height)
        img_size: 图像尺寸 (width, height)
        
    Returns:
        list: 像素坐标 [x0, y0, x1, y1]
    """
    pdf_width, pdf_height = pdf_size
    img_width, img_height = img_size
    
    scale_x = img_width / pdf_width
    scale_y = img_height / pdf_height
    
    x0 = int(pdf_coord[0] * scale_x)
    y0 = int(pdf_coord[1] * scale_y)
    x1 = int(pdf_coord[2] * scale_x)
    y1 = int(pdf_coord[3] * scale_y)
    
    # 确保坐标在图像范围内
    x0 = max(0, min(x0, img_width-1))
    y0 = max(0, min(y0, img_height-1))
    x1 = max(0, min(x1, img_width-1))
    y1 = max(0, min(y1, img_height-1))
    
    return [x0, y0, x1, y1]


def convert_coordinates(coords, source_size, target_size):
    """
    在不同尺寸之间转换坐标
    
    Args:
        coords: 原始坐标 [x0, y0, x1, y1]
        source_size: 源图像尺寸 (width, height)
        target_size: 目标图像尺寸 (width, height)
        
    Returns:
        list: 转换后的坐标 [x0, y0, x1, y1]
    """
    src_w, src_h = source_size
    tgt_w, tgt_h = target_size
    
    scale_x = tgt_w / src_w
    scale_y = tgt_h / src_h
    
    x0 = int(coords[0] * scale_x)
    y0 = int(coords[1] * scale_y)
    x1 = int(coords[2] * scale_x)
    y1 = int(coords[3] * scale_y)
    
    return [x0, y0, x1, y1]