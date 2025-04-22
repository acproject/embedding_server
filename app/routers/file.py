import time
import io  # 添加缺少的io模块导入
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, Request
from loguru import logger
import numpy as np
import traceback


router = APIRouter()

@router.post("/api/pdf_embedding")
async def pdf_embedding_endpoint(
    request: Request,
    file: UploadFile = File(...),
    include_layout: bool = Query(False, description="是否包含布局信息"),
    include_formulas: bool = Query(False, description="是否包含公式信息")
):
    # 获取服务实例
    pdf_service = request.app.state.pdf_service
    embedding_service = request.app.state.text_embedder
    
    if not pdf_service or not embedding_service:
        raise HTTPException(status_code=503, detail="服务暂时不可用")

    logger.info(f"收到 /api/pdf_embedding 请求，文件名: {file.filename}, 包含布局信息: {include_layout}, 包含公式信息: {include_formulas}")

    pdf_content = await file.read()
    start_time = time.time()

    try:
        # 1. 调用 PDF 服务处理 PDF 并获取 Markdown 和布局信息
        logger.info(f"调用 pdf_service.convert_pdf_to_markdown")
        markdown_text, layout_info = pdf_service.convert_pdf_to_markdown(pdf_content)
        logger.info(f"PDF 转换完成，耗时: {time.time() - start_time:.2f} 秒")

        # 检查 Markdown 是否为空或包含错误信息
        if not markdown_text or markdown_text == "*无法提取PDF内容*":
            logger.warning(f"无法从 PDF 文件 {file.filename} 提取内容")
            # 即使无法提取内容，也可能需要返回空 embedding 或错误
            embedding = np.zeros(embedding_service.get_embedding_dimension()).tolist()
            model_name = embedding_service.text_model_name
        else:
            # 2. 使用获取到的 Markdown 文本生成嵌入向量
            logger.info("使用 Markdown 文本生成嵌入向量")
            raw_embedding = embedding_service.get_embedding(markdown_text, content_type="text")
            model_name = embedding_service.text_model_name # 获取文本模型名称

            # 确保 embedding 是 NumPy 数组
            if not isinstance(raw_embedding, np.ndarray):
                logger.error(f"从 embedding 服务接收到非预期的类型: {type(raw_embedding)}")
                raise HTTPException(status_code=500, detail="内部错误：嵌入服务返回格式错误")

            # 展平并转换为列表
            embedding = raw_embedding.flatten().tolist()

        # 3. 构建响应
        response = {
            "embedding": embedding,
            "model": model_name,
            "dimensions": len(embedding),
            "processing_time": time.time() - start_time
        }
        
        # 4. 根据请求参数添加额外信息
        if include_layout:
            # 添加布局信息
            response["layout"] = layout_info
        
        if include_formulas and "formulas" in layout_info:
            # 仅添加公式信息
            response["formulas"] = layout_info["formulas"]
        
        return response
        
    except Exception as e:
        logger.error(f"处理 PDF 文件时出错: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"处理 PDF 文件时出错: {str(e)}")

@router.post("/api/pdf_layout")
async def pdf_layout_endpoint(
    request: Request,
    file: UploadFile = File(...),
    page: Optional[int] = Query(None, description="指定页码，从1开始，不指定则返回所有页面")
):
    """
    分析PDF文档的布局，返回布局信息
    """
    # 获取服务实例
    pdf_service = request.app.state.pdf_service
    
    if not pdf_service:
        raise HTTPException(status_code=503, detail="服务暂时不可用")

    logger.info(f"收到 /api/pdf_layout 请求，文件名: {file.filename}, 页码: {page}")

    pdf_content = await file.read()
    start_time = time.time()

    try:
        # 调用 PDF 服务处理 PDF 并获取布局信息
        _, layout_info = pdf_service.convert_pdf_to_markdown(pdf_content)
        
        # 如果指定了页码，只返回该页的布局信息
        if page is not None:
            if page < 1 or page > len(layout_info["pages"]):
                raise HTTPException(status_code=400, detail=f"页码超出范围: {page}, 总页数: {len(layout_info['pages'])}")
            
            return {
                "page": layout_info["pages"][page-1],
                "processing_time": time.time() - start_time
            }
        
        # 返回所有页面的布局信息
        return {
            "layout": layout_info,
            "processing_time": time.time() - start_time
        }
        
    except Exception as e:
        logger.error(f"分析 PDF 布局时出错: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"分析 PDF 布局时出错: {str(e)}")

@router.post("/api/formula_recognition")
async def formula_recognition_endpoint(
    request: Request,
    file: UploadFile = File(...),
    x1: int = Query(..., description="裁剪区域左上角x坐标"),
    y1: int = Query(..., description="裁剪区域左上角y坐标"),
    x2: int = Query(..., description="裁剪区域右下角x坐标"),
    y2: int = Query(..., description="裁剪区域右下角y坐标"),
    page: int = Query(1, description="页码，从1开始")
):
    """
    识别PDF文档中指定区域的公式
    """
    # 获取服务实例
    pdf_service = request.app.state.pdf_service
    
    if not pdf_service:
        raise HTTPException(status_code=503, detail="服务暂时不可用")

    logger.info(f"收到 /api/formula_recognition 请求，文件名: {file.filename}, 区域: ({x1}, {y1}, {x2}, {y2}), 页码: {page}")

    pdf_content = await file.read()
    start_time = time.time()

    try:
        import pdfplumber
        from PIL import Image
        
        # 打开PDF文件
        pdf_file = io.BytesIO(pdf_content)
        with pdfplumber.open(pdf_file) as pdf:
            if page < 1 or page > len(pdf.pages):
                raise HTTPException(status_code=400, detail=f"页码超出范围: {page}, 总页数: {len(pdf.pages)}")
            
            # 获取指定页面
            pdf_page = pdf.pages[page-1]
            
            # 提取页面图像
            img = pdf_page.to_image(resolution=300).original
            
            # 检查坐标是否有效
            if x1 < 0 or y1 < 0 or x2 > img.width or y2 > img.height or x1 >= x2 or y1 >= y2:
                raise HTTPException(status_code=400, detail=f"无效的坐标: ({x1}, {y1}, {x2}, {y2}), 图像尺寸: {img.width}x{img.height}")
            
            # 裁剪指定区域
            crop_img = img.crop((x1, y1, x2, y2))
            crop_img_np = np.array(crop_img)
            
            # 识别公式
            formula = pdf_service.formula_extractor.recognize_formula(crop_img_np)
            
            # 转换图像为base64
            buffered = io.BytesIO()
            crop_img.save(buffered, format="PNG")
            img_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
            
            # 返回结果
            return {
                "formula": formula,
                "image_base64": img_base64,
                "coordinates": [x1, y1, x2, y2],
                "page": page,
                "processing_time": time.time() - start_time
            }
            
    except Exception as e:
        logger.error(f"识别公式时出错: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"识别公式时出错: {str(e)}")

@router.post("/api/image_formula_recognition")
async def image_formula_recognition_endpoint(
    request: Request,
    file: UploadFile = File(...)
):
    """
    识别图像中的公式
    """
    # 获取服务实例
    pdf_service = request.app.state.pdf_service
    
    if not pdf_service:
        raise HTTPException(status_code=503, detail="服务暂时不可用")

    logger.info(f"收到 /api/image_formula_recognition 请求，文件名: {file.filename}")

    image_content = await file.read()
    start_time = time.time()

    try:
        from PIL import Image
        import io
        import base64  # 添加base64模块导入
        
        # 打开图像
        img = Image.open(io.BytesIO(image_content))
        img_np = np.array(img)
        
        # 识别公式
        formula = pdf_service.formula_extractor.recognize_formula(img_np)
        
        # 转换图像为base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
        
        # 返回结果
        return {
            "formula": formula,
            "image_base64": img_base64,
            "processing_time": time.time() - start_time
        }
            
    except Exception as e:
        detailed_error = traceback.format_exc() # <--- 获取详细 traceback
        logger.error(f"处理 PDF 嵌入时出错: {e}\n{detailed_error}") # <--- 在日志中也记录详细错误
        # 注意：在生产环境中暴露详细错误信息可能存在安全风险
        raise HTTPException(status_code=500, detail=f"PDF 处理失败: {str(e)}\nTraceback:\n{detailed_error}") # <--- 将 traceback 添加到 detail


@router.post("/api/pix2text_formula_recognition")
async def pix2text_formula_recognition_endpoint(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Query("formula", description="识别模式: formula, text, table 或 auto")
):
    """
    使用 Pix2Text 识别图像中的公式、文本或表格
    """
    logger.info(f"收到 /api/pix2text_formula_recognition 请求，文件名: {file.filename}, 模式: {mode}")
    
    image_content = await file.read()
    start_time = time.time()
    
    try:
        from PIL import Image
        import io
        import base64
        import torch  # Add this import
        from pix2text import Pix2Text
        
        # 初始化 Pix2Text
        p2t = Pix2Text(
            model_dir="./models",  # 使用本地模型目录
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # 打开图像
        img = Image.open(io.BytesIO(image_content))
        
        # 根据模式选择识别方法
        if mode == "formula":
            # 识别公式
            result = p2t.recognize_formula(img)
        elif mode == "text":
            # 识别文本
            result = p2t.recognize(img)
        elif mode == "table":
            # 识别表格
            result = p2t.recognize_table(img)
        elif mode == "auto":
            # 自动识别
            result = p2t.recognize_all(img)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的识别模式: {mode}")
        
        # 转换图像为base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"
        
        # 构建响应
        response = {
            "result": result,
            "mode": mode,
            "image_base64": img_base64,
            "processing_time": time.time() - start_time
        }
        
        # 如果是自动模式，返回更详细的结果
        if mode == "auto" and isinstance(result, dict):
            response["formulas"] = result.get("formulas", [])
            response["texts"] = result.get("texts", [])
            response["tables"] = result.get("tables", [])
        
        return response
        
    except ImportError as e:
        logger.error(f"缺少必要的库: {e}")
        raise HTTPException(status_code=500, detail=f"服务器缺少必要的库: {str(e)}")
    except Exception as e:
        logger.error(f"使用 Pix2Text 识别图像时出错: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"识别图像时出错: {str(e)}")