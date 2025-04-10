#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例Python客户端代码，展示如何调用嵌入服务
"""

import requests
import json
import numpy as np

# 嵌入服务URL
EMBEDDING_SERVICE_URL = "http://localhost:8086/api/embedding"

def get_embedding(text):
    """
    从嵌入服务获取文本的向量表示
    
    Args:
        text: 需要向量化的文本
        
    Returns:
        numpy.ndarray: 文本的向量表示
    """
    try:
        # 准备请求数据
        payload = {"text": text}
        headers = {"Content-Type": "application/json"}
        
        # 发送POST请求
        response = requests.post(EMBEDDING_SERVICE_URL, 
                               data=json.dumps(payload), 
                               headers=headers)
        
        # 检查响应状态
        response.raise_for_status()
        
        # 解析响应数据
        result = response.json()
        
        if "embedding" in result:
            # 将列表转换为numpy数组
            embedding = np.array(result["embedding"], dtype=np.float32)
            return embedding
        else:
            raise ValueError("响应中没有找到embedding字段")
            
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        raise
    except ValueError as e:
        print(f"解析响应失败: {e}")
        raise
    except Exception as e:
        print(f"发生未知错误: {e}")
        raise

def main():
    # 示例文本
    text = "这是一个测试文本，用于演示如何获取嵌入向量"
    
    try:
        # 获取嵌入向量
        embedding = get_embedding(text)
        
        # 打印结果
        print(f"文本: {text}")
        print(f"向量维度: {embedding.shape}")
        print(f"向量前5个值: {embedding[:5]}")
        
        # 这里可以添加将向量存储到PostgreSQL的代码
        # ...
        
    except Exception as e:
        print(f"处理失败: {e}")

if __name__ == "__main__":
    main()