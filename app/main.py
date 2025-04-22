from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
import os
import time
import base64
from typing import Optional, Union, List
from dotenv import load_dotenv
import torch
import sys

# 设置使用镜像站点（移到这里，确保在导入其他模块前设置）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from app.services.multimodal_embedding_service import MultimodalEmbeddingService
from app.services.pdf import PDFService
# 导入路由器
from app.routers import pdf_services
from app.routers import file  # 添加导入file路由器
from app.routers import embedding  # 添加导入embedding路由器

# 加载环境变量
load_dotenv()

# 创建FastAPI应用
app = FastAPI(
    title="多模态嵌入服务",
    description="支持文本、PDF和图像的嵌入向量生成服务",
    version="2.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由器
app.include_router(pdf_services.router, prefix="/api/pdf_services")
app.include_router(file.router)  # 添加file路由器注册
app.include_router(embedding.router)  # 添加embedding路由器注册

# 初始化嵌入服务
embedding_service = None

# 请求模型
class TextRequest(BaseModel):
    text: str
    content_type: Optional[str] = "text"

class Base64ImageRequest(BaseModel):
    image: str  # Base64编码的图像
    content_type: str = "image"

# 响应模型
class EmbeddingResponse(BaseModel):
    embedding: list[float]
    dimensions: int
    model: str
    processing_time: float
    content_type: str
    visualization_images: Optional[List[str]] = None  # 添加可视化图像列表字段

@app.on_event("startup")
async def startup_event():
    try:
        # 修改为使用本地模型路径
        text_model_name = "models/LaBSE"
        image_model_name = "models/vector/CLIP"
        logger.info(f"正在加载模型: {text_model_name}, {image_model_name}")
        
        global embedding_service
        # 使用正确的参数初始化多模态嵌入服务
        # 注意：先不要在MultimodalEmbeddingService中初始化PDFService
        embedding_service = MultimodalEmbeddingService(
            text_model_name, 
            image_model_name,
            init_pdf_service=False  # 添加参数，防止内部初始化PDFService
        )
        
        # 初始化 PDF 服务并设置到嵌入服务中
        pdf_service = PDFService(
            models_dir="./models",
            text_embedder=embedding_service,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # 将PDF服务设置到嵌入服务中
        embedding_service.pdf_service = pdf_service
        
        # 将服务实例添加到应用状态
        app.state.pdf_service = pdf_service
        app.state.text_embedder = embedding_service
        
        logger.info("服务启动成功")
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise e

@app.post("/api/embedding", response_model=EmbeddingResponse)
async def get_embedding(request: TextRequest):
    if not embedding_service:
        raise HTTPException(status_code=503, detail="嵌入服务尚未初始化")
    
    if not request.text or request.text.strip() == "":
        raise HTTPException(status_code=400, detail="文本不能为空")
    
    try:
        start_time = time.time()
        embedding = embedding_service.get_embedding(request.text, request.content_type)
        processing_time = time.time() - start_time
        
        return {
            "embedding": embedding.tolist(),
            "dimensions": len(embedding),
            "model": embedding_service.text_model_name if request.content_type == "text" else embedding_service.image_model_name,
            "processing_time": processing_time,
            "content_type": request.content_type
        }
    except Exception as e:
        logger.error(f"生成嵌入向量时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成嵌入向量失败: {str(e)}")

@app.post("/api/image_embedding", response_model=EmbeddingResponse)
async def get_image_embedding(request: Base64ImageRequest):
    if not embedding_service:
        raise HTTPException(status_code=503, detail="嵌入服务尚未初始化")
    
    if not request.image or not request.image.startswith("data:image"):
        raise HTTPException(status_code=400, detail="图像格式不正确，需要base64编码的图像")
    
    try:
        start_time = time.time()
        embedding = embedding_service.get_embedding(request.image, "image")
        processing_time = time.time() - start_time
        
        return {
            "embedding": embedding.tolist(),
            "dimensions": len(embedding),
            "model": embedding_service.image_model_name,
            "processing_time": processing_time,
            "content_type": "image"
        }
    except Exception as e:
        logger.error(f"生成图像嵌入向量时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成图像嵌入向量失败: {str(e)}")

@app.post("/api/pdf_embedding", response_model=EmbeddingResponse)
async def get_pdf_embedding(
    file: UploadFile = File(...),
    visualize: bool = Query(False, description="是否启用PDF处理可视化")
):
    if not embedding_service:
        raise HTTPException(status_code=503, detail="嵌入服务尚未初始化")
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持PDF文件")
    
    try:
        # 读取上传的PDF文件内容
        pdf_content = await file.read()
        
        start_time = time.time()
        # 确保传递正确的参数
        result = embedding_service.get_embedding(pdf_content, "pdf", visualize=visualize)
        
        # 根据visualize参数解析返回值
        if visualize:
            embedding, visualization_images = result
        else:
            embedding = result
            visualization_images = None
            
        processing_time = time.time() - start_time
        
        response = {
            "embedding": embedding.tolist(),
            "dimensions": len(embedding),
            "model": embedding_service.text_model_name,  # PDF转换为文本后使用文本模型
            "processing_time": processing_time,
            "content_type": "pdf"
        }
        
        # 如果启用了可视化，添加可视化图像到响应
        if visualize and visualization_images:
            response["visualization_images"] = visualization_images
            
        return response
    except Exception as e:
        logger.error(f"生成PDF嵌入向量时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成PDF嵌入向量失败: {str(e)}")

@app.get("/health")
async def health_check():
    if embedding_service:
        return {
            "status": "healthy", 
            "text_model": embedding_service.text_model_name,
            "image_model": embedding_service.image_model_name
        }
    return {"status": "initializing"}

if __name__ == "__main__":
    import uvicorn
    # 配置 Loguru
    logger.remove()
    logger.add(sys.stderr, level="INFO") # 输出到 stderr，级别为 INFO
    logger.add("file_{time}.log", rotation="500 MB", level="DEBUG") # 输出到文件，级别为 DEBUG
    uvicorn.run(
        "app.main:app", 
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8086)),
        reload=True
    )