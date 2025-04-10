# 文本嵌入服务 (Text Embedding Service)

这是一个基于FastAPI的服务，使用sentence-transformers/LaBSE模型将文本转换为向量表示。该服务提供REST API接口，可以被Java等其他服务调用，用于实现RAG（检索增强生成）系统。

## 功能

- 使用sentence-transformers/LaBSE模型将文本转换为向量
- 提供REST API接口，返回文本的向量表示
- 支持与Java服务集成，便于将向量存储到PostgreSQL数据库

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

1. 启动服务

```bash
uvicorn app.main:app --reload
```

2. API调用示例

```bash
curl -X POST "http://localhost:8000/api/embedding" \
     -H "Content-Type: application/json" \
     -d '{"text": "这是一个测试文本"}'
```

## API文档

启动服务后，访问 http://localhost:8000/docs 查看完整的API文档。

## 与Java服务集成

服务的响应格式与提供的Java代码兼容，可以直接通过Java的RestTemplate调用并解析响应。