from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
import os
import time
from dotenv import load_dotenv

# 设置使用镜像站点（移到这里，确保在导入其他模块前设置）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from app.services.embedding_service import EmbeddingService

# 加载环境变量
load_dotenv()

# 创建FastAPI应用
app = FastAPI(
    title="文本嵌入服务",
    description="使用sentence-transformers/LaBSE模型将文本转换为向量表示",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化嵌入服务
embedding_service = None

# 请求模型
class TextRequest(BaseModel):
    text: str

# 响应模型
class EmbeddingResponse(BaseModel):
    embedding: list[float]
    dimensions: int
    model: str
    processing_time: float

@app.on_event("startup")
async def startup_event():
    try:
        # 修改为使用本地模型路径
        model_name = "/Users/acproject/workspace/python_projects/embedding_server/models/LaBSE"
        logger.info(f"正在加载模型: {model_name}")
        global embedding_service
        embedding_service = EmbeddingService(model_name)
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
        embedding = embedding_service.get_embedding(request.text)
        processing_time = time.time() - start_time
        
        return {
            "embedding": embedding.tolist(),
            "dimensions": len(embedding),
            "model": embedding_service.model_name,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"生成嵌入向量时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成嵌入向量失败: {str(e)}")

@app.get("/health")
async def health_check():
    if embedding_service:
        return {"status": "healthy", "model": embedding_service.model_name}
    return {"status": "initializing"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app", 
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8086)),
        reload=True
    )