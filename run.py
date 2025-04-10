import os
import uvicorn
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

if __name__ == "__main__":
    # 从环境变量获取配置，如果没有则使用默认值
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"启动嵌入服务，访问地址: http://{host}:{port}")
    print(f"API文档地址: http://{host}:{port}/docs")
    
    # 启动服务
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True
    )