from fastapi import APIRouter, File, UploadFile, Request, Body, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import io
import numpy as np
from PIL import Image
import base64
import time

router = APIRouter(
    tags=["PDF服务"],
    responses={404: {"description": "服务不可用"}},
)

# 布局分析请求模型
class LayoutAnalysisRequest(BaseModel):
    image_base64: str
    
# 公式识别请求模型
class FormulaRecognitionRequest(BaseModel):
    image_base64: str

# 布局分析服务
@router.post("/layout_analysis")
async def analyze_layout(request: Request, file: UploadFile = File(...), page_num: int = 0):
    """
    分析PDF页面的布局，返回布局元素列表
    
    Args:
        file: PDF文件
        page_num: 页码 (从0开始)
        
    Returns:
        List[Dict]: 布局元素列表，每个元素包含类型、坐标和置信度
    """
    try:
        # 获取布局分析器
        layout_analyzer = request.app.state.pdf_service.layout_analyzer
        
        # 读取PDF文件
        pdf_content = await file.read()
        
        # 使用pdfplumber打开PDF
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            if page_num >= len(pdf.pages):
                raise HTTPException(status_code=400, detail=f"页码超出范围: {page_num}, 总页数: {len(pdf.pages)}")
                
            # 获取指定页面
            page = pdf.pages[page_num]
            
            # 提取页面图像
            img = page.to_image(resolution=300).original
            img_np = np.array(img)
            
            # 分析布局
            layout_results = layout_analyzer.analyze_page(img_np)
            
            # 直接返回布局结果
            return layout_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"布局分析失败: {str(e)}")

# 公式识别服务
@router.post("/formula_recognition")
async def recognize_formula(request: Request, data: FormulaRecognitionRequest):
    """
    识别图像中的公式
    
    Args:
        data: 包含base64编码图像的请求
        
    Returns:
        Dict: 包含识别结果的响应
    """
    try:
        # 获取公式提取器
        formula_extractor = request.app.state.pdf_service.formula_extractor
        
        # 解码base64图像
        image_data = base64.b64decode(data.image_base64.split(",")[-1])
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # 识别公式
        start_time = time.time()
        formula_result = formula_extractor.recognize_formula(image_np)
        processing_time = time.time() - start_time
        
        # 检查结果是否为base64编码的图像
        if formula_result.startswith("iVBOR") or formula_result.startswith("R0lGOD") or formula_result.startswith("/9j/"):
            return {
                "result_type": "image",
                "formula_image": formula_result,
                "processing_time": processing_time
            }
        else:
            return {
                "result_type": "latex",
                "formula_latex": formula_result,
                "processing_time": processing_time
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"公式识别失败: {str(e)}")

# PDF页面处理服务
@router.post("/process_page")
async def process_page(request: Request, file: UploadFile = File(...), page_num: int = 0):
    """
    处理PDF页面，返回布局信息和内容
    
    Args:
        file: PDF文件
        page_num: 页码 (从0开始)
        
    Returns:
        Dict: 包含页面处理结果的响应
    """
    try:
        # 获取PDF服务
        pdf_service = request.app.state.pdf_service
        
        # 读取PDF文件
        pdf_content = await file.read()
        
        # 使用pdfplumber打开PDF
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            if page_num >= len(pdf.pages):
                raise HTTPException(status_code=400, detail=f"页码超出范围: {page_num}, 总页数: {len(pdf.pages)}")
                
            # 获取指定页面
            page = pdf.pages[page_num]
            
            # 提取页面图像
            img = page.to_image(resolution=300).original
            img_np = np.array(img)
            
            # 分析布局
            start_time = time.time()
            layout = pdf_service.layout_analyzer.analyze_page(img_np)
            
            # 处理每个布局元素
            elements = []
            for block_idx, block in enumerate(layout):
                block_type = block["type"]
                x1, y1, x2, y2 = block["coordinates"]
                
                try:
                    # 裁剪区域
                    crop_img = img.crop((x1, y1, x2, y2))
                    crop_img_np = np.array(crop_img)
                    
                    # 根据类型处理不同元素
                    element = {
                        "id": block_idx,
                        "type": block_type,
                        "coordinates": [x1, y1, x2, y2],
                        "confidence": block.get("confidence", 1.0)
                    }
                    
                    # 转换图像为base64
                    buffered = io.BytesIO()
                    crop_img.save(buffered, format="PNG")
                    img_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
                    element["image"] = img_base64
                    
                    # 根据类型提取内容
                    if block_type == "text":
                        # 提取文本
                        text = pdf_service.text_extractor.extract_text(crop_img_np)
                        element["content"] = text
                    elif block_type == "formula":
                        # 识别公式
                        formula = pdf_service.formula_extractor.recognize_formula(crop_img_np)
                        element["content"] = formula
                    elif block_type == "table":
                        # 提取表格
                        table_data = pdf_service.table_extractor.extract_table(crop_img_np)
                        element["content"] = table_data
                    elif block_type == "figure":
                        # 图像不需要额外处理，已经有base64了
                        element["content"] = "图像"
                    
                    elements.append(element)
                except Exception as e:
                    # 处理单个元素失败，记录错误但继续处理其他元素
                    elements.append({
                        "id": block_idx,
                        "type": block_type,
                        "coordinates": [x1, y1, x2, y2],
                        "error": str(e)
                    })
            
            # 转换整页图像为base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            page_img_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
            
            # 提取页面文本（使用pdfplumber的文本提取）
            page_text = page.extract_text() or ""
            
            processing_time = time.time() - start_time
            
            # 返回结果
            return {
                "page_number": page_num,
                "page_image": page_img_base64,
                "page_text": page_text,
                "elements": elements,
                "page_size": [img.width, img.height],
                "processing_time": processing_time
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理PDF页面失败: {str(e)}")

# 添加PDF文本提取服务
@router.post("/extract_text")
async def extract_text(request: Request, file: UploadFile = File(...), page_num: Optional[int] = None):
    """
    从PDF文件中提取文本
    
    Args:
        file: PDF文件
        page_num: 可选的页码，不提供则提取所有页面
        
    Returns:
        Dict: 包含提取的文本
    """
    try:
        # 读取PDF文件
        pdf_content = await file.read()
        
        # 使用pdfplumber打开PDF
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            start_time = time.time()
            
            if page_num is not None:
                # 提取指定页面的文本
                if page_num < 0 or page_num >= len(pdf.pages):
                    raise HTTPException(status_code=400, detail=f"页码超出范围: {page_num}, 总页数: {len(pdf.pages)}")
                
                page_text = pdf.pages[page_num].extract_text() or ""
                result = {
                    "page_number": page_num,
                    "text": page_text,
                    "processing_time": time.time() - start_time
                }
            else:
                # 提取所有页面的文本
                all_text = []
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    all_text.append({
                        "page_number": i,
                        "text": page_text
                    })
                
                result = {
                    "pages": all_text,
                    "total_pages": len(pdf.pages),
                    "processing_time": time.time() - start_time
                }
            
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提取PDF文本失败: {str(e)}")

# 添加分析整个PDF布局的服务
@router.post("/layout_analysis_all")
async def analyze_layout_all(request: Request, file: UploadFile = File(...)):
    """
    分析整个PDF文档的布局，返回所有页面的布局元素列表
    
    Args:
        file: PDF文件
        
    Returns:
        Dict: 包含所有页面布局元素的响应
    """
    try:
        # 获取布局分析器
        layout_analyzer = request.app.state.pdf_service.layout_analyzer
        
        # 读取PDF文件
        pdf_content = await file.read()
        
        # 使用pdfplumber打开PDF
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            total_pages = len(pdf.pages)
            
            # 分析所有页面的布局
            start_time = time.time()
            pages_layout = []
            
            for page_num in range(total_pages):
                # 获取指定页面
                page = pdf.pages[page_num]
                
                # 提取页面图像
                img = page.to_image(resolution=300).original
                img_np = np.array(img)
                
                # 分析布局
                layout_results = layout_analyzer.analyze_page(img_np)
                
                # 添加页码信息
                pages_layout.append({
                    "page_number": page_num,
                    "page_size": [img.width, img.height],
                    "layout": layout_results
                })
            
            processing_time = time.time() - start_time
            
            # 返回结果
            return {
                "total_pages": total_pages,
                "pages_layout": pages_layout,
                "processing_time": processing_time
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"布局分析失败: {str(e)}")

# 添加处理整个PDF的服务
@router.post("/process_document")
async def process_document(request: Request, file: UploadFile = File(...)):
    """
    处理整个PDF文档，返回所有页面的布局信息和内容
    
    Args:
        file: PDF文件
        
    Returns:
        Dict: 包含所有页面处理结果的响应
    """
    try:
        # 获取PDF服务
        pdf_service = request.app.state.pdf_service
        
        # 读取PDF文件
        pdf_content = await file.read()
        
        # 使用pdfplumber打开PDF
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            total_pages = len(pdf.pages)
            
            # 处理所有页面
            start_time = time.time()
            pages = []
            
            for page_num in range(total_pages):
                # 获取指定页面
                page = pdf.pages[page_num]
                
                # 提取页面图像
                img = page.to_image(resolution=300).original
                img_np = np.array(img)
                
                # 分析布局
                layout = pdf_service.layout_analyzer.analyze_page(img_np)
                
                # 处理每个布局元素
                elements = []
                for block_idx, block in enumerate(layout):
                    block_type = block["type"]
                    x1, y1, x2, y2 = block["coordinates"]
                    
                    try:
                        # 裁剪区域
                        crop_img = img.crop((x1, y1, x2, y2))
                        crop_img_np = np.array(crop_img)
                        
                        # 根据类型处理不同元素
                        element = {
                            "id": block_idx,
                            "type": block_type,
                            "coordinates": [x1, y1, x2, y2],
                            "confidence": block.get("confidence", 1.0)
                        }
                        
                        # 转换图像为base64
                        buffered = io.BytesIO()
                        crop_img.save(buffered, format="PNG")
                        img_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
                        element["image"] = img_base64
                        
                        # 根据类型提取内容
                        if block_type == "text":
                            # 提取文本
                            text = pdf_service.text_extractor.extract_text(crop_img_np)
                            element["content"] = text
                        elif block_type == "formula":
                            # 识别公式
                            formula = pdf_service.formula_extractor.recognize_formula(crop_img_np)
                            element["content"] = formula
                        elif block_type == "table":
                            # 提取表格
                            table_data = pdf_service.table_extractor.extract_table(crop_img_np)
                            element["content"] = table_data
                        elif block_type == "figure":
                            # 图像不需要额外处理，已经有base64了
                            element["content"] = "图像"
                        
                        elements.append(element)
                    except Exception as e:
                        # 处理单个元素失败，记录错误但继续处理其他元素
                        elements.append({
                            "id": block_idx,
                            "type": block_type,
                            "coordinates": [x1, y1, x2, y2],
                            "error": str(e)
                        })
                
                # 转换整页图像为base64
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                page_img_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
                
                # 提取页面文本（使用pdfplumber的文本提取）
                page_text = page.extract_text() or ""
                
                # 添加页面信息到结果中
                pages.append({
                    "page_number": page_num,
                    "page_image": page_img_base64,
                    "page_text": page_text,
                    "elements": elements,
                    "page_size": [img.width, img.height]
                })
            
            processing_time = time.time() - start_time
            
            # 提取整个文档的文本
            all_text = "\n\n".join([page.get("page_text", "") for page in pages])
            
            # 返回结果
            return {
                "total_pages": total_pages,
                "pages": pages,
                "document_text": all_text,
                "processing_time": processing_time
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理PDF文档失败: {str(e)}")