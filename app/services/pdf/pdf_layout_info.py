class PDFLayoutInfo:
    """PDF布局信息存储结构"""
    
    def __init__(self, pdf_path=None, pdf_content=None):
        """初始化PDF布局信息"""
        self.pdf_path = pdf_path
        self.pdf_content = pdf_content
        self.pages = []  # 存储所有页面的布局信息
        self.original_sizes = []  # 存储原始PDF页面尺寸
        self.model_sizes = []  # 存储模型处理的尺寸
        self.rendered_sizes = []  # 存储渲染后的图像尺寸
        self.formulas = {}  # 存储公式信息，格式: {页码: [公式列表]}
        self.tables = {}  # 存储表格信息，格式: {页码: [表格列表]}
        self.figures = {}  # 存储图片信息，格式: {页码: [图片列表]}
        
    def add_page(self, page_idx, layout_elements, rendered_size, model_size, original_size=None):
        """添加页面布局信息"""
        # 确保页面索引有效
        while len(self.pages) <= page_idx:
            self.pages.append([])
            self.original_sizes.append(None)
            self.model_sizes.append(None)
            self.rendered_sizes.append(None)
            
        # 存储布局元素和尺寸信息
        self.pages[page_idx] = layout_elements
        self.rendered_sizes[page_idx] = rendered_size
        self.model_sizes[page_idx] = model_size
        if original_size:
            self.original_sizes[page_idx] = original_size
        
    def get_page_layout(self, page_idx):
        """获取指定页面的布局信息"""
        if 0 <= page_idx < len(self.pages):
            return self.pages[page_idx]
        return []
    
    def get_element_by_type(self, page_idx, element_type):
        """获取指定页面中指定类型的所有元素"""
        if 0 <= page_idx < len(self.pages):
            return [elem for elem in self.pages[page_idx] if elem["type"].lower() == element_type.lower()]
        return []
    
    def map_coordinates(self, page_idx, coords, from_model_to_rendered=True):
        """
        映射坐标系统
        
        Args:
            page_idx: 页面索引
            coords: 坐标元组 (x0, y0, x1, y1)
            from_model_to_rendered: 如果为True，从模型坐标映射到渲染坐标；否则反之
            
        Returns:
            映射后的坐标元组
        """
        if 0 <= page_idx < len(self.pages):
            rendered_size = self.rendered_sizes[page_idx]
            model_size = self.model_sizes[page_idx]
            
            if not rendered_size or not model_size:
                return coords
            
            x0, y0, x1, y1 = coords
            
            # 添加偏移校正
            offset_x = int(rendered_size[0])  # 水平偏移校正，约5%
            offset_y = int(rendered_size[1])  # 垂直偏移校正，约5%
            
            if from_model_to_rendered:
                # 从模型坐标映射到渲染坐标
                x_ratio = rendered_size[0] / model_size[0]
                y_ratio = rendered_size[1] / model_size[1]
                
                mapped_x0 = max(0, int(x0 * x_ratio) - offset_x)
                mapped_y0 = max(0, int(y0 * y_ratio) - offset_y)
                mapped_x1 = max(mapped_x0 + 1, min(int(x1 * x_ratio) - offset_x, rendered_size[0]))
                mapped_y1 = max(mapped_y0 + 1, min(int(y1 * y_ratio) - offset_y, rendered_size[1]))
            else:
                # 从渲染坐标映射到模型坐标
                x_ratio = model_size[0] / rendered_size[0]
                y_ratio = model_size[1] / rendered_size[1]
                
                mapped_x0 = int((x0 + offset_x) * x_ratio)
                mapped_y0 = int((y0 + offset_y) * y_ratio)
                mapped_x1 = int((x1 + offset_x) * x_ratio)
                mapped_y1 = int((y1 + offset_y) * y_ratio)
                
            return (mapped_x0, mapped_y0, mapped_x1, mapped_y1)
        
        return coords  # 如果页面索引无效，返回原始坐标
    
    def get_cropped_element(self, page_idx, element_idx, image=None, use_original_coords=True):
        """
        从页面图像中裁剪出指定元素
        
        Args:
            page_idx: 页面索引
            element_idx: 元素索引
            image: 页面图像(PIL Image或numpy数组)，如果为None则需要重新渲染
            use_original_coords: 是否使用原始坐标
            
        Returns:
            裁剪后的图像
        """
        if 0 <= page_idx < len(self.pages) and 0 <= element_idx < len(self.pages[page_idx]):
            element = self.pages[page_idx][element_idx]
            
            # 获取坐标
            if use_original_coords and "original_coordinates" in element:
                coords = element["original_coordinates"]
            else:
                coords = element["coordinates"]
                # 如果需要，转换坐标
                if use_original_coords:
                    coords = self.map_coordinates(page_idx, coords)
                    
            # 确保坐标有效
            x0, y0, x1, y1 = [int(c) for c in coords]
            
            # 如果没有提供图像，尝试重新渲染
            if image is None and self.pdf_content:
                doc = fitz.open(stream=self.pdf_content, filetype="pdf")
                if 0 <= page_idx < len(doc):
                    page = doc[page_idx]
                    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
            
            # 裁剪图像
            if image is not None:
                if isinstance(image, np.ndarray):
                    if y1 <= image.shape[0] and x1 <= image.shape[1]:
                        return image[y0:y1, x0:x1]
                else:
                    if y1 <= image.height and x1 <= image.width:
                        return image.crop((x0, y0, x1, y1))
                
        return None
    
    def add_formula(self, page_idx, formula_info):
        """添加公式信息"""
        if page_idx not in self.formulas:
            self.formulas[page_idx] = []
        self.formulas[page_idx].append(formula_info)
    
    def add_table(self, page_idx, table_info):
        """添加表格信息"""
        if page_idx not in self.tables:
            self.tables[page_idx] = []
        self.tables[page_idx].append(table_info)
    
    def add_figure(self, page_idx, figure_info):
        """添加图片信息"""
        if page_idx not in self.figures:
            self.figures[page_idx] = []
        self.figures[page_idx].append(figure_info)
    
    def get_formulas(self, page_idx=None):
        """获取公式信息"""
        if page_idx is not None:
            return self.formulas.get(page_idx, [])
        return self.formulas
    
    def get_tables(self, page_idx=None):
        """获取表格信息"""
        if page_idx is not None:
            return self.tables.get(page_idx, [])
        return self.tables
    
    def get_figures(self, page_idx=None):
        """获取图片信息"""
        if page_idx is not None:
            return self.figures.get(page_idx, [])
        return self.figures
    
    def to_markdown(self, page_idx=None):
        """将PDF内容转换为Markdown格式"""
        markdown = ""
        
        # 处理指定页面或所有页面
        pages_to_process = [page_idx] if page_idx is not None else range(len(self.pages))
        
        for idx in pages_to_process:
            if 0 <= idx < len(self.pages):
                page_markdown = f"## 第 {idx+1} 页\n\n"
                
                # 按照元素在页面中的位置排序（从上到下）
                elements = sorted(self.pages[idx], key=lambda x: x["coordinates"][1])
                
                for element in elements:
                    element_type = element["type"].lower()
                    
                    if element_type == "title":
                        if "extracted_text" in element:
                            page_markdown += f"### {element['extracted_text']}\n\n"
                    
                    elif element_type == "plain text":
                        if "extracted_text" in element:
                            page_markdown += f"{element['extracted_text']}\n\n"
                    
                    elif element_type == "isolate_formula":
                        formula_idx = element.get("formula_idx")
                        if formula_idx is not None and idx in self.formulas:
                            for formula in self.formulas[idx]:
                                if formula.get("element_idx") == formula_idx:
                                    latex = formula.get("latex", "")
                                    page_markdown += f"$$\n{latex}\n$$\n\n"
                                    break
                    
                    elif element_type == "table":
                        table_idx = element.get("table_idx")
                        if table_idx is not None and idx in self.tables:
                            for table in self.tables[idx]:
                                if table.get("element_idx") == table_idx:
                                    table_md = table.get("markdown", "")
                                    page_markdown += f"{table_md}\n\n"
                                    break
                    
                    elif element_type == "figure":
                        figure_idx = element.get("figure_idx")
                        if figure_idx is not None and idx in self.figures:
                            for figure in self.figures[idx]:
                                if figure.get("element_idx") == figure_idx:
                                    img_base64 = figure.get("base64", "")
                                    caption = figure.get("caption", "图片")
                                    page_markdown += f"![{caption}](data:image/png;base64,{img_base64})\n\n"
                                    break
                
                markdown += page_markdown
                
                # 添加页面分隔符
                if idx < len(self.pages) - 1:
                    markdown += "---\n\n"
        
        return markdown