#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例Python客户端代码，展示如何调用嵌入服务
"""

import os
import requests
import json
import numpy as np

# 嵌入服务URL
EMBEDDING_SERVICE_URL = "http://localhost:8086/api/embedding"

def get_embedding(content, content_type="text"):
    """
    从嵌入服务获取内容的向量表示
    
    Args:
        content: 需要向量化的内容（文本字符串、PDF或图像文件路径）
        content_type: 内容类型，可选值："text"、"pdf"、"image"
        
    Returns:
        numpy.ndarray: 内容的向量表示
    """
    try:
        # 准备请求数据
        payload = {}
        headers = {"Content-Type": "application/json"}
        
        # 根据内容类型处理数据
        if content_type == "text":
            payload = {
                "text": content,
                "content_type": content_type
            }
        elif content_type == "pdf":
            # 读取PDF文件内容
            try:
                with open(content, "rb") as f:
                    pdf_content = f.read()
                # 直接发送PDF二进制内容，不进行Base64编码
                payload = {
                    "text": pdf_content,
                    "content_type": content_type
                }
            except FileNotFoundError:
                raise ValueError(f"PDF文件不存在: {content}")
            except Exception as e:
                raise ValueError(f"读取PDF文件失败: {e}")
        elif content_type == "image":
            # 读取图像文件内容
            try:
                with open(content, "rb") as f:
                    image_content = f.read()
                # 将图像内容转换为Base64编码
                import base64
                image_base64 = base64.b64encode(image_content).decode("utf-8")
                payload = {
                    "text": f"data:image/jpeg;base64,{image_base64}",
                    "content_type": content_type
                }
            except FileNotFoundError:
                raise ValueError(f"图像文件不存在: {content}")
            except Exception as e:
                raise ValueError(f"读取图像文件失败: {e}")
        
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
            print(f"模型: {result['model']}")
            print(f"处理时间: {result['processing_time']:.2f}秒")
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
    try:
        # 1. 文本示例
        text = "这是一个测试文本，用于演示如何获取嵌入向量"
        text_embedding = get_embedding(text, "text")
        print("\n=== 文本处理结果 ===")
        print(f"文本: {text}")
        print(f"向量维度: {text_embedding.shape}")
        print(f"向量前5个值: {text_embedding[:5]}")
        
        # 2. PDF文档示例
        pdf_path = "examples/2024-Aligning Large Language Models with Humans.pdf"
        # 使用专门的PDF API端点处理PDF文件
        try:
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
            
            # 使用multipart/form-data发送PDF文件
            files = {"file": (os.path.basename(pdf_path), pdf_content, "application/pdf")}
            response = requests.post("http://localhost:8086/api/pdf_embedding", files=files)
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应数据
            result = response.json()
            
            if "embedding" in result:
                # 将列表转换为numpy数组
                pdf_embedding = np.array(result["embedding"], dtype=np.float32)
                print("\n=== PDF处理结果 ===")
                print(f"文件: {pdf_path}")
                print(f"向量维度: {pdf_embedding.shape}")
                print(f"向量前5个值: {pdf_embedding[:5]}")
                print(f"模型: {result['model']}")
                print(f"处理时间: {result['processing_time']:.2f}秒")
                
                # 设置输出目录
                out_dir = os.path.join(os.path.dirname(pdf_path), 'out')
                os.makedirs(out_dir, exist_ok=True)

                # 生成对应的Markdown文件
                if result.get('markdown_text'):
                    markdown_path = os.path.join(out_dir, os.path.basename(os.path.splitext(pdf_path)[0] + '.md'))
                    try:
                        with open(markdown_path, 'w', encoding='utf-8') as f:
                            f.write(result['markdown_text'])
                        print(f"已生成Markdown文件: {markdown_path}")
                    except Exception as e:
                        print(f"保存Markdown文件失败: {e}")
                else:
                    print("警告: 服务端返回的markdown_text为空")

                # 保存向量数据到CSV文件
                csv_path = os.path.join(out_dir, os.path.basename(os.path.splitext(pdf_path)[0] + '_vector.csv'))
                try:
                    np.savetxt(csv_path, pdf_embedding, delimiter=',')
                    print(f"已保存向量数据到: {csv_path}")
                except Exception as e:
                    print(f"保存CSV文件失败: {e}")
            else:
                print(f"PDF处理失败: 响应中没有找到embedding字段")
        except Exception as e:
            print(f"PDF处理失败: {e}")
        
        # 3. 图像示例
        # image_path = "path/to/image.jpg"
        # image_embedding = get_embedding(image_path, "image")
        # print("\n=== 图像处理结果 ===")
        # print(f"文件: {image_path}")
        # print(f"向量维度: {image_embedding.shape}")
        # print(f"向量前5个值: {image_embedding[:5]}")
        
    except Exception as e:
        print(f"处理失败: {e}")

if __name__ == "__main__":
    main()