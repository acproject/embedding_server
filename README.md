# 文本嵌入服务 (Text Embedding Service)

这是一个基于FastAPI的服务，提供多模态内容的向量表示功能。支持文本、PDF和图像的向量化，可用于构建高效的RAG（检索增强生成）系统。

## 功能特性

- 文本向量化：使用sentence-transformers/LaBSE模型
- PDF处理：支持PDF文档的文本提取和向量化
- 图像处理：支持图像的向量表示
- RESTful API：提供统一的API接口
- 多语言支持：提供Python和Java客户端示例
- 高性能：异步处理，支持批量请求

## 安装部署

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 下载模型（首次运行需要）
```bash
python download_model.py
```

3. 启动服务
```bash
# 开发环境
uvicorn app.main:app --host 0.0.0.0 --port 8086 --reload

# 生产环境
gunicorn app.main:app -c gunicorn_conf.py
```

## API使用示例

1. 文本向量化
```bash
curl -X POST "http://localhost:8086/api/embedding" \
     -H "Content-Type: application/json" \
     -d '{"text": "这是一个测试文本", "content_type": "text"}'
```

2. PDF文档处理
```bash
curl -X POST "http://localhost:8086/api/embedding" \
     -H "Content-Type: application/json" \
     -d '{"text": "path/to/document.pdf", "content_type": "pdf"}'
```

3. 图像处理
```bash
curl -X POST "http://localhost:8086/api/embedding" \
     -H "Content-Type: application/json" \
     -d '{"text": "path/to/image.jpg", "content_type": "image"}'
```

## API文档

启动服务后，访问 http://localhost:8086/docs 查看完整的API文档。

## 客户端集成

- Python客户端示例：查看 `examples/python_client_example.py`
- Java客户端：支持通过RestTemplate调用并解析响应

## Docker部署

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d
```