# 4. 增强版PDF布局分析函数，包含公式、表格和图片解析
def analyze_pdf_with_elements(pdf_content, max_pages=200):
    """
    分析PDF布局并解析公式、表格和图片
    
    Args:
        pdf_content: PDF文档的二进制内容
        max_pages: 最大处理页数
        
    Returns:
        PDFLayoutInfo对象，包含布局和元素信息
    """
    # 初始化各种分析器
    layout_analyzer = LayoutAnalyzer(models_dir=os.path.join(project_root, "models"), device="cpu")
    text_extractor = TextExtractor(models_dir=os.path.join(project_root, "models"), device="cpu")
    formula_extractor = FormulaExtractor(models_dir=os.path.join(project_root, "models"), device="cpu")
    table_extractor = TableExtractor(models_dir=os.path.join(project_root, "models"), device="cpu")
    visualizer = PDFVisualization()
    
    # 初始化布局信息存储结构
    layout_info = PDFLayoutInfo(pdf_content=pdf_content)
    
    # 打开PDF
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    total_pages = len(doc)
    print(f"PDF总页数: {total_pages}")
    
    # 处理所有页面（或最多max_pages页）
    pages_to_process = min(max_pages, total_pages)
    
    for page_idx in range(pages_to_process):
        print(f"\n处理第 {page_idx+1}/{pages_to_process} 页")
        page = doc[page_idx]
        
        # 获取原始页面尺寸
        original_size = (page.rect.width, page.rect.height)
        print(f"原始页面尺寸: {original_size}")
        
        # 渲染页面为图像
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 使用2x缩放获得更清晰的图像
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        img_np = np.array(img)
        
        
        # 获取渲染后的图像尺寸
        rendered_size = (img.width, img.height)
        print(f"渲染后图像尺寸: {rendered_size}")
        
        # 分析布局
        layout_results = layout_analyzer.analyze_page(img_np)
        print(f"第 {page_idx+1} 页检测到的布局元素数量: {len(layout_results)}")
        
        # 获取模型处理的尺寸
        model_size = (800, 1024)
        if hasattr(layout_analyzer, 'model_input_size'):
            model_size = layout_analyzer.model_input_size
        
        # 存储布局信息
        layout_info.add_page(page_idx, layout_results, rendered_size, model_size, original_size)
        
        # 创建可视化图像
        layout_img = img.copy()
        draw = ImageDraw.Draw(layout_img)
        
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
        }
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("Arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        print(f"第 {page_idx+1} 页检测到的元素类型:")
        for i, block in enumerate(layout_results):
            
            block_type = block["type"].lower()  # 转为小写以匹配颜色字典
            print(f"{i+1}. {block_type}")
            
            # 获取模型坐标
            x0, y0, x1, y1 = block["coordinates"]
            
            # 映射到渲染坐标
            # mapped_coords = layout_info.map_coordinates(page_idx, model_coords)
            # x0, y0, x1, y1 = mapped_coords
            
            # 更新block中的坐标为映射后的坐标
            # block["original_coordinates"] = mapped_coords
            mapped_coords = block["coordinates"]
            # 绘制边界框
            line_width = max(3, int(min(rendered_size) / 300))
            color = colors.get(block_type, (0, 0, 200))
            draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)
            
            # 添加标签
            
            confidence = block.get("confidence", 0)
            label = f"{i+1}.{block_type} ({confidence:.2f})"
            text_bbox = draw.textbbox((x0, y0), label, font=font)
            draw.rectangle(text_bbox, fill=(255, 255, 255, 200))
            draw.text((x0, y0-15), label, fill=color, font=font)
            
            # 裁剪区域
            try:
                crop_img = img.crop((x0, y0, x1, y1))
                crop_img_np = np.array(crop_img)
            except Exception as e:
                print(f"  裁剪失败: {e}")
                continue
            
            # 根据元素类型进行不同处理
            if block_type in ["plain text", "title"]:
                # 提取文本
                try:
                    extracted_text = text_extractor.extract_text(crop_img_np)
                    if extracted_text:
                        block["extracted_text"] = extracted_text
                        print(f"  文本: {extracted_text[:100]}{'...' if len(extracted_text) > 100 else ''}")
                except Exception as e:
                    print(f"  文本提取失败: {e}")
            
            elif block_type == "isolate_formula":
                # 提取公式
                try:
                    latex = formula_extractor.recognize_formula(crop_img_np)
                    if latex:
                        formula_info = {
                            "element_idx": i,
                            "coordinates": mapped_coords,
                            "latex": latex,
                            "confidence": confidence
                        }
                        layout_info.add_formula(page_idx, formula_info)
                        block["formula_idx"] = i
                        print(f"  公式: {latex[:100]}{'...' if len(latex) > 100 else ''}")
                except Exception as e:
                    print(f"  公式提取失败: {e}")
            
            elif block_type == "table":
                # 提取表格
                try:
                    table_markdown = table_extractor.extract_table(crop_img_np)
                    if not table_markdown:
                        # 如果高级表格识别失败，回退到基本方法
                        table_markdown = table_extractor.extract_table_basic(page, mapped_coords)
                    
                    if table_markdown:
                        table_info = {
                            "element_idx": i,
                            "coordinates": mapped_coords,
                            "markdown": table_markdown,
                            "confidence": confidence
                        }
                        layout_info.add_table(page_idx, table_info)
                        block["table_idx"] = i
                        print(f"  表格: {table_markdown[:100]}{'...' if len(table_markdown) > 100 else ''}")
                except Exception as e:
                    print(f"  表格提取失败: {e}")
            
            elif block_type == "figure":
                # 处理图片
                try:
                    img_base64 = visualizer.image_to_base64(crop_img)
                    figure_info = {
                        "element_idx": i,
                        "coordinates": mapped_coords,
                        "base64": img_base64,
                        "confidence": confidence,
                        "caption": "图片"
                    }
                    
                    # 尝试提取图片中的文本作为可能的标题
                    try:
                        img_text = text_extractor.extract_text(crop_img_np)
                        if img_text:
                            figure_info["caption"] = img_text
                    except:
                        pass
                    
                    layout_info.add_figure(page_idx, figure_info)
                    block["figure_idx"] = i
                    print(f"  图片已提取")
                except Exception as e:
                    print(f"  图片处理失败: {e}")
        
        # 显示带有布局标注的图像
        plt.figure(figsize=(15, 20))
        plt.imshow(layout_img)
        plt.title(f"第 {page_idx+1} 页布局分析", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # 显示提取的内容统计
        print("\n提取内容统计:")
        print(f"  文本块: {len([b for b in layout_results if b['type'].lower() in ['plain text', 'title']])}")
        print(f"  公式: {len(layout_info.get_formulas(page_idx))}")
        print(f"  表格: {len(layout_info.get_tables(page_idx))}")
        print(f"  图片: {len(layout_info.get_figures(page_idx))}")
    
    return layout_info