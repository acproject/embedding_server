from fastapi import APIRouter, Depends, File, UploadFile, Request, Query
from typing import Optional, List
import io
from pydantic import BaseModel

router = APIRouter()

# 添加用于文本嵌入的请求模型
class TextEmbeddingRequest(BaseModel):
    text: str

@router.post("/embed/pdf")
async def embed_pdf(
    request: Request,
    file: UploadFile = File(...),
    visualize: bool = Query(False, description="是否返回可视化结果")
):
    # 读取上传的PDF文件
    pdf_content = await file.read()
    
    # 获取PDF服务实例
    pdf_service = request.app.state.pdf_service
    
    # 只处理PDF获取嵌入向量
    embedding = pdf_service.process_pdf(pdf_content)
    return {
        "embedding": embedding.tolist()
    }

# 添加新的路由用于文本嵌入
@router.post("/embedding")
async def embed_text(
    request: Request,
    text_request: TextEmbeddingRequest
):
    """
    获取文本的嵌入向量
    
    Args:
        text_request: 包含要嵌入的文本的请求体
        
    Returns:
        dict: 包含嵌入向量的响应
    """
    # 获取嵌入服务实例
    embedding_service = request.app.state.embedding_service
    
    # 使用嵌入服务获取文本的嵌入向量
    try:
        embedding = embedding_service.get_embedding(text_request.text)
        return {
            "embedding": embedding.tolist(),
            "model": embedding_service.model_name,
            "dimensions": embedding.shape[0]
        }
    except Exception as e:
        return {
            "error": f"生成嵌入向量时出错: {str(e)}"
        }