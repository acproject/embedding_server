from fastapi import APIRouter, Depends, File, UploadFile, Request, Query
from typing import Optional, List
import io

router = APIRouter()

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
    
    # 处理PDF
    if visualize:
        # 处理PDF并获取可视化结果
        embedding, visualization_images = pdf_service.process_pdf_with_visualization(pdf_content)
        return {
            "embedding": embedding.tolist(),
            "visualization": visualization_images
        }
    else:
        # 只处理PDF获取嵌入向量
        embedding = pdf_service.process_pdf(pdf_content)
        return {
            "embedding": embedding.tolist()
        }