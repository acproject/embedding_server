import time
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from loguru import logger
import numpy as np
import traceback # <--- 添加 traceback 导入

# 导入服务
from app.services.pdf_service import PDFService
from app.services.multimodal_embedding_service import MultimodalEmbeddingService

router = APIRouter()

# 初始化服务实例 (假设在应用启动时创建并传递，或者在这里创建)
# 注意：在生产环境中，依赖注入是更好的方式
try:
    pdf_service_instance = PDFService()
    # 假设 MultimodalEmbeddingService 用于获取最终的 embedding
    embedding_service_instance = MultimodalEmbeddingService()
except Exception as e:
    logger.error(f"服务初始化失败: {e}")
    # 如果服务无法初始化，可以阻止应用启动或返回错误
    pdf_service_instance = None
    embedding_service_instance = None

@router.post("/api/pdf_embedding")
async def pdf_embedding_endpoint(
    file: UploadFile = File(...),
    visualize: bool = Query(False, description="是否生成并返回可视化图像")
):
    if not pdf_service_instance or not embedding_service_instance:
        raise HTTPException(status_code=503, detail="服务暂时不可用")

    logger.info(f"收到 /api/pdf_embedding 请求，文件名: {file.filename}, 是否可视化: {visualize}")

    pdf_content = await file.read()
    start_time = time.time()

    try:
        # 1. 调用 PDF 服务处理 PDF 并获取 Markdown 和可视化结果
        logger.info(f"调用 pdf_service.convert_pdf_to_markdown，visualize={visualize}")
        markdown_text, visualization_results = pdf_service_instance.convert_pdf_to_markdown(
            pdf_content, visualize=visualize
        )
        logger.info(f"PDF 转换完成，耗时: {time.time() - start_time:.2f} 秒")

        # 检查 Markdown 是否为空或包含错误信息
        if not markdown_text or markdown_text == "*无法提取PDF内容*":
            logger.warning(f"无法从 PDF 文件 {file.filename} 提取内容")
            # 即使无法提取内容，也可能需要返回空 embedding 或错误
            # 这里我们选择返回错误，但也可以返回空 embedding
            # raise HTTPException(status_code=400, detail="无法从PDF提取有效内容")
            # 或者返回一个表示失败的 embedding (例如全零向量)
            embedding = np.zeros(embedding_service_instance.text_model.get_sentence_embedding_dimension()).tolist()
            model_name = embedding_service_instance.text_model_name
        else:
            # 2. 使用获取到的 Markdown 文本生成嵌入向量
            logger.info("使用 Markdown 文本生成嵌入向量")
            raw_embedding = embedding_service_instance.get_embedding(markdown_text, content_type="text")
            model_name = embedding_service_instance.text_model_name # 获取文本模型名称

            # 确保 embedding 是 NumPy 数组
            if not isinstance(raw_embedding, np.ndarray):
                logger.error(f"从 embedding 服务接收到非预期的类型: {type(raw_embedding)}")
                raise HTTPException(status_code=500, detail="内部错误：嵌入服务返回格式错误")

            # 展平并转换为列表
            embedding_list = raw_embedding.flatten().tolist()

            # 添加日志记录以检查转换前的 embedding
            logger.debug(f"Embedding before float conversion - Type: {type(embedding_list)}, Value: {embedding_list[:10]}...") # Log only first 10 elements

            # 确保列表中的所有元素都是 float 类型
            embedding: List[float] = [] # 明确类型
            try:
                embedding = [float(x) for x in embedding_list]
            except (ValueError, TypeError) as cast_error:
                logger.error(f"无法将 embedding 元素转换为 float: {cast_error}. Embedding list sample: {embedding_list[:10]}...")
                # 根据需要处理错误，例如返回默认值或引发异常
                raise HTTPException(status_code=500, detail=f"内部错误：无法处理嵌入向量格式")

        processing_time = time.time() - start_time
        logger.info(f"总处理时间: {processing_time:.2f} 秒")

        # 3. 构建响应
        response_data = {
            "embedding": embedding,
            "markdown_text": markdown_text, # 返回 Markdown 文本
            "model": model_name,
            "processing_time": processing_time
        }

        # 如果请求了可视化，则添加可视化结果
        if visualize:
            response_data["visualization_images"] = visualization_results
        else:
            # 明确表示没有可视化结果，或者不包含该字段
            response_data["visualization_images"] = []

        return response_data

    except Exception as e:
        detailed_error = traceback.format_exc() # <--- 获取详细 traceback
        logger.error(f"处理 PDF 嵌入时出错: {e}\n{detailed_error}") # <--- 在日志中也记录详细错误
        # 注意：在生产环境中暴露详细错误信息可能存在安全风险
        raise HTTPException(status_code=500, detail=f"PDF 处理失败: {str(e)}\nTraceback:\n{detailed_error}") # <--- 将 traceback 添加到 detail