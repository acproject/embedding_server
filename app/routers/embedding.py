from fastapi import APIRouter, Depends, File, UploadFile, Request, Query
from typing import Optional, List
import io
from pydantic import BaseModel

router = APIRouter()

# 添加用于文本嵌入的请求模型
class TextEmbeddingRequest(BaseModel):
    text: str

# 添加用于图片嵌入的请求模型
class ImageEmbeddingRequest(BaseModel):
    image_base64: str  # Base64编码的图片数据

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
    # 获取嵌入服务实例 - 修改这一行
    embedding_service = request.app.state.text_embedder
    
    # 使用嵌入服务获取文本的嵌入向量
    try:
        embedding = embedding_service.get_embedding(text_request.text)
        return {
            "embedding": embedding.tolist(),
            "model": embedding_service.text_model_name,
            "dimensions": len(embedding),
            "content_type": "text"
        }
    except Exception as e:
        return {
            "error": f"生成嵌入向量时出错: {str(e)}"
        }

# 添加新的路由用于图片嵌入
@router.post("/image_embedding")
async def embed_image(
    request: Request,
    file: UploadFile = File(...)
):
    """
    获取图片的嵌入向量
    
    Args:
        file: 上传的图片文件
        
    Returns:
        dict: 包含嵌入向量的响应
    """
    # 获取嵌入服务实例
    embedding_service = request.app.state.text_embedder
    
    try:
        # 读取上传的图片文件内容
        image_content = await file.read()
        
        # 使用嵌入服务获取图片的嵌入向量
        embedding = embedding_service.get_embedding(image_content, content_type="image")
        
        return {
            "embedding": embedding.tolist(),
            "model": embedding_service.image_model_name,
            "dimensions": len(embedding),
            "content_type": "image"
        }
    except Exception as e:
        return {
            "error": f"生成图片嵌入向量时出错: {str(e)}"
        }

# 添加新的路由用于Base64编码的图片嵌入
@router.post("/base64_image_embedding")
async def embed_base64_image(
    request: Request,
    image_request: ImageEmbeddingRequest
):
    """
    获取Base64编码图片的嵌入向量
    
    Args:
        image_request: 包含Base64编码图片的请求体
        
    Returns:
        dict: 包含嵌入向量的响应
    """
    # 获取嵌入服务实例
    embedding_service = request.app.state.text_embedder
    
    try:
        # 使用嵌入服务获取图片的嵌入向量
        embedding = embedding_service.get_embedding(image_request.image_base64, content_type="image")
        
        return {
            "embedding": embedding.tolist(),
            "model": embedding_service.image_model_name,
            "dimensions": len(embedding),
            "content_type": "image"
        }
    except Exception as e:
        return {
            "error": f"生成图片嵌入向量时出错: {str(e)}"
        }