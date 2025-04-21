#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例Python客户端代码，展示如何调用嵌入服务
"""

import os
import sys
import requests
import json
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import fitz  # PyMuPDF
import logging
from pathlib import Path

# 添加项目根目录到系统路径
project_root = "/Users/acproject/workspace/python_projects/embedding_server"
sys.path.append(project_root)

# 导入相关服务
try:
    from app.services.pdf_service import PDFService
    from app.services.pdf.core import PDFService as PDFCoreService
    from app.services.pdf.formula import FormulaExtractor
    from app.services.pdf.layout import LayoutAnalyzer
    from app.services.pdf.text import TextExtractor
    from app.services.pdf.table import TableExtractor
    from app.services.pdf.visualization import PDFVisualization
    from app.services.pdf.pdf_layout_info import PDFLayoutInfo
except ImportError:
    print("警告: 无法导入PDF服务模块，将使用API方式处理PDF")

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 嵌入服务URL
EMBEDDING_SERVICE_URL = "http://localhost:8086/api/embedding"
PDF_EMBEDDING_URL = "http://localhost:8086/api/pdf_embedding"

def analyze_pdf_with_elements(pdf_content, max_pages=200):
    """
    分析PDF布局并解析公式、表格和图片
    
    Args:
        pdf_content: PDF文档的二进制内容
        max_pages: 最大处理页数
        
    Returns:
        PDFLayoutInfo对象，包含布局和元素信息
    """
    # 初始化各种分析器
    layout_analyzer = LayoutAnalyzer(models_dir=os.path.join(project_root, "models"), device="cpu")
    text_extractor = TextExtractor(models_dir=os.path.join(project_root, "models"), device="cpu")
    formula_extractor = FormulaExtractor(models_dir=os.path.join(project_root, "models"), device="cpu")
    table_extractor = TableExtractor(models_dir=os.path.join(project_root, "models"), device="cpu")
    visualizer = PDFVisualization()
    
    # 初始化布局信息存储结构
    layout_info = PDFLayoutInfo(pdf_content=pdf_content)
    
    # 打开PDF
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    total_pages = len(doc)
    print(f"PDF总页数: {total_pages}")
    
    # 处理所有页面（或最多max_pages页）
    pages_to_process = min(max_pages, total_pages)
    
    for page_idx in range(pages_to_process):
        print(f"\n处理第 {page_idx+1}/{pages_to_process} 页")
        page = doc[page_idx]
        
        # 获取原始页面尺寸
        original_size = (page.rect.width, page.rect.height)
        print(f"原始页面尺寸: {original_size}")
        
        # 渲染页面为图像
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 使用2x缩放获得更清晰的图像
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        img_np = np.array(img)
        
        
        # 获取渲染后的图像尺寸
        rendered_size = (img.width, img.height)
        print(f"渲染后图像尺寸: {rendered_size}")
        
        # 分析布局
        layout_results = layout_analyzer.analyze_page(img_np)
        print(f"第 {page_idx+1} 页检测到的布局元素数量: {len(layout_results)}")
        
        # 获取模型处理的尺寸
        model_size = (800, 1024)
        if hasattr(layout_analyzer, 'model_input_size'):
            model_size = layout_analyzer.model_input_size
        
        # 存储布局信息
        layout_info.add_page(page_idx, layout_results, rendered_size, model_size, original_size)
        
        # 创建可视化图像
        layout_img = img.copy()
        draw = ImageDraw.Draw(layout_img)
        
        # 为不同类型的元素使用不同颜色
        colors = {
            "plain text": (0, 200, 0),      # 绿色
            "title": (255, 0, 0),           # 红色
            "figure": (0, 0, 255),          # 蓝色
            "table": (255, 165, 0),         # 橙色
            "table_caption": (255, 135, 0), # 深橙色
            "table_footnote": (255, 100, 0),# 红橙色
            "isolate_formula": (128, 0, 128), # 紫色
            "list": (0, 255, 255),          # 青色
            "figure_caption": (0, 20, 45),  # 深蓝色
            "abandon": (100, 100, 100),     # 灰色
        }
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("Arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        print(f"第 {page_idx+1} 页检测到的元素类型:")
        for i, block in enumerate(layout_results):
            
            block_type = block["type"].lower()  # 转为小写以匹配颜色字典
            print(f"{i+1}. {block_type}")
            
            # 获取模型坐标
            x0, y0, x1, y1 = block["coordinates"]
            
            # 映射到渲染坐标
            # mapped_coords = layout_info.map_coordinates(page_idx, model_coords)
            # x0, y0, x1, y1 = mapped_coords
            
            # 更新block中的坐标为映射后的坐标
            # block["original_coordinates"] = mapped_coords
            mapped_coords = block["coordinates"]
            # 绘制边界框
            line_width = max(3, int(min(rendered_size) / 300))
            color = colors.get(block_type, (0, 0, 200))
            draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)
            
            # 添加标签
            
            confidence = block.get("confidence", 0)
            label = f"{i+1}.{block_type} ({confidence:.2f})"
            text_bbox = draw.textbbox((x0, y0), label, font=font)
            draw.rectangle(text_bbox, fill=(255, 255, 255, 200))
            draw.text((x0, y0-15), label, fill=color, font=font)
            
            # 裁剪区域
            try:
                crop_img = img.crop((x0, y0, x1, y1))
                crop_img_np = np.array(crop_img)
            except Exception as e:
                print(f"  裁剪失败: {e}")
                continue
            
            # 根据元素类型进行不同处理
            if block_type in ["plain text", "title"]:
                # 提取文本
                try:
                    extracted_text = text_extractor.extract_text(crop_img_np)
                    if extracted_text:
                        block["extracted_text"] = extracted_text
                        print(f"  文本: {extracted_text[:100]}{'...' if len(extracted_text) > 100 else ''}")
                except Exception as e:
                    print(f"  文本提取失败: {e}")
            
            elif block_type == "isolate_formula":
                # 提取公式
                try:
                    latex = formula_extractor.recognize_formula(crop_img_np)
                    if latex:
                        formula_info = {
                            "element_idx": i,
                            "coordinates": mapped_coords,
                            "latex": latex,
                            "confidence": confidence
                        }
                        layout_info.add_formula(page_idx, formula_info)
                        block["formula_idx"] = i
                        print(f"  公式: {latex[:100]}{'...' if len(latex) > 100 else ''}")
                except Exception as e:
                    print(f"  公式提取失败: {e}")
            
            elif block_type == "table":
                # 提取表格
                try:
                    table_markdown = table_extractor.extract_table(crop_img_np)
                    if table_markdown:
                        table_info = {
                            "element_idx": i,
                            "coordinates": mapped_coords,
                            "markdown": table_markdown,
                            "confidence": confidence
                        }
                        layout_info.add_table(page_idx, table_info)
                        block["table_idx"] = i
                        print(f"  表格: {table_markdown[:100]}{'...' if len(table_markdown) > 100 else ''}")
                except Exception as e:
                    print(f"  表格提取失败: {e}")
            
            elif block_type == "figure":
                # 处理图片
                try:
                    figure_info = {
                        "element_idx": i,
                        "coordinates": mapped_coords,
                        "confidence": confidence
                    }
                    layout_info.add_figure(page_idx, figure_info)
                    block["figure_idx"] = i
                    print(f"  图片: 尺寸 {crop_img.width}x{crop_img.height}")
                except Exception as e:
                    print(f"  图片处理失败: {e}")
        
        # 生成页面的Markdown文本
        try:
            page_text = text_extractor.extract_text_from_page(page)
            if hasattr(layout_info, 'add_page_text'):
                layout_info.add_page_text(page_idx, page_text)
        except Exception as e:
            print(f"页面文本提取失败: {e}")
    
    # 生成完整的Markdown文本
    try:
        markdown_text = text_extractor.extract_text_to_markdown(doc)
        layout_info.markdown_text = markdown_text
    except Exception as e:
        print(f"Markdown生成失败: {e}")
        layout_info.markdown_text = "Markdown生成失败"
    
    return layout_info

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

def process_pdf_local(pdf_path, visualize=True):
    """
    使用本地PDF处理pipeline处理PDF文件
    
    Args:
        pdf_path: PDF文件路径
        visualize: 是否生成可视化结果
        
    Returns:
        tuple: (embedding, layout_info, visualization_images)
    """
    try:
        # 读取PDF文件
        with open(pdf_path, "rb") as f:
            pdf_content = f.read()
        
        # 使用管道分析PDF
        layout_info = analyze_pdf_with_elements(pdf_content)
        
        # 创建PDF服务实例获取嵌入向量
        pdf_service = PDFCoreService()
        embedding = pdf_service.get_embedding(pdf_content)
        
        # 创建可视化工具
        visualizer = PDFVisualization()
        
        # 存储可视化图像
        visualization_images = []
        
        if visualize:
            # 打开PDF文档
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            
            # 处理每一页
            for page_idx in range(min(len(doc), 10)):  # 限制处理前10页
                print(f"生成第 {page_idx+1} 页可视化...")
                
                # 获取页面
                page = doc[page_idx]
                
                # 渲染页面为图像
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                page_image = Image.open(io.BytesIO(img_data))
                
                # 可视化页面布局
                layout_viz = visualizer.visualize_page_layout(page_image, page)
                visualization_images.append(layout_viz)
                
                # 可视化布局分析结果
                layout_elements = layout_info.get_page_layout(page_idx)
                if layout_elements:
                    layout_analysis_viz = visualizer.visualize_layout(page_image, layout_elements)
                    visualization_images.append(layout_analysis_viz)
                
                # 可视化表格
                tables = layout_info.get_tables(page_idx)
                if tables:
                    table_viz = visualizer.visualize_table_detection(page_image, tables)
                    visualization_images.append(table_viz)
                
                # 可视化公式
                formulas = layout_info.get_formulas(page_idx)
                if formulas:
                    formula_viz = visualizer.visualize_formula_detection(page_image, {"detection": {"boxes": formulas}})
                    visualization_images.append(formula_viz)
            
            # 调整图像大小到800*1024范围内
            if visualization_images:
                resized_images = []
                for img in visualization_images:
                    resized_img = visualizer.resize_image_to_fit(img, 800, 1024)
                    resized_images.append(resized_img)
                visualization_images = resized_images
        
        return embedding, layout_info, visualization_images
    
    except Exception as e:
        logger.error(f"本地处理PDF失败: {e}", exc_info=True)
        raise

def process_pdf_api(pdf_path, visualize=True):
    """
    使用API处理PDF文件
    
    Args:
        pdf_path: PDF文件路径
        visualize: 是否生成可视化结果
        
    Returns:
        tuple: (embedding, markdown_text, visualization_images)
    """
    try:
        # 读取PDF文件
        with open(pdf_path, "rb") as f:
            pdf_content = f.read()
        
        # 使用multipart/form-data发送PDF文件
        files = {"file": (os.path.basename(pdf_path), pdf_content, "application/pdf")}
        
        # 添加可视化参数
        params = {}
        if visualize:
            params["visualize"] = "true"
        
        # 发送请求
        response = requests.post(PDF_EMBEDDING_URL, files=files, params=params)
        
        # 检查响应状态
        response.raise_for_status()
        
        # 解析响应数据
        result = response.json()
        
        if "embedding" in result:
            # 将列表转换为numpy数组
            embedding = np.array(result["embedding"], dtype=np.float32)
            
            # 获取Markdown文本
            markdown_text = result.get("markdown_text", "")
            
            # 获取可视化图像
            visualization_images = []
            if visualize and "visualization_images" in result:
                for img_base64 in result["visualization_images"]:
                    try:
                        # 从base64字符串解码图像
                        if img_base64.startswith('data:image'):
                            img_data = img_base64.split(',')[1]
                        else:
                            img_data = img_base64
                            
                        img_bytes = base64.b64decode(img_data)
                        img = Image.open(io.BytesIO(img_bytes))
                        visualization_images.append(img)
                    except Exception as e:
                        logger.error(f"解码可视化图像失败: {e}")
            
            return embedding, markdown_text, visualization_images
        else:
            raise ValueError("响应中没有找到embedding字段")
    
    except requests.exceptions.HTTPError as e:
        logger.error(f"API处理PDF失败: {e}")
        raise
    except Exception as e:
        logger.error(f"API处理PDF失败: {e}", exc_info=True)
        raise

def save_visualization_images(images, output_dir):
    """
    保存可视化图像
    
    Args:
        images: 图像列表
        output_dir: 输出目录
        
    Returns:
        list: 保存的图像路径列表
    """
    image_paths = []
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存每张图像
    for i, img in enumerate(images):
        img_path = os.path.join(output_dir, f"visualization_{i+1}.png")
        img.save(img_path)
        image_paths.append(img_path)
        print(f"已保存可视化图像: {img_path}")
        print(f"图像 {i+1} 尺寸: {img.width}x{img.height}")
    
    return image_paths

def main():
    try:
        # 设置PDF文件路径
        pdf_path = os.path.join(project_root, "examples/2024-Aligning Large Language Models with Humans.pdf")
        
        # 设置输出目录
        out_dir = os.path.join(os.path.dirname(pdf_path), 'out')
        os.makedirs(out_dir, exist_ok=True)
        
        # 设置可视化输出目录
        viz_dir = os.path.join(out_dir, 'visualization')
        
        # 是否生成可视化结果
        visualize = True
        
        # 尝试使用本地处理pipeline
        try:
            print("\n=== 使用本地处理pipeline ===")
            embedding, layout_info, visualization_images = process_pdf_local(pdf_path, visualize)
            
            # 输出处理结果
            print("\n=== PDF处理结果 ===")
            print(f"文件: {pdf_path}")
            print(f"向量维度: {embedding.shape}")
            print(f"向量前5个值: {embedding[:5]}")
            
            # 保存Markdown文件
            markdown_path = os.path.join(out_dir, os.path.basename(os.path.splitext(pdf_path)[0] + '.md'))
            try:
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(layout_info.markdown_text)
                print(f"已生成Markdown文件: {markdown_path}")
            except Exception as e:
                print(f"保存Markdown文件失败: {e}")
            
            # 保存向量数据到CSV文件
            csv_path = os.path.join(out_dir, os.path.basename(os.path.splitext(pdf_path)[0] + '_vector.csv'))
            try:
                np.savetxt(csv_path, embedding, delimiter=',')
                print(f"已保存向量数据到: {csv_path}")
            except Exception as e:
                print(f"保存CSV文件失败: {e}")
            
            # 保存可视化图像
            if visualize and visualization_images:
                print(f"\n=== 可视化结果 ===")
                print(f"共有 {len(visualization_images)} 张可视化图像")
                save_visualization_images(visualization_images, viz_dir)
        
        except (ImportError, NameError) as e:
            print(f"本地处理pipeline不可用: {e}")
            print("尝试使用API方式处理PDF...")
            
            # 使用API处理PDF
            embedding, markdown_text, visualization_images = process_pdf_api(pdf_path, visualize)
            
            # 输出处理结果
            print("\n=== PDF处理结果（API方式）===")
            print(f"文件: {pdf_path}")
            print(f"向量维度: {embedding.shape}")
            print(f"向量前5个值: {embedding[:5]}")
            
            # 保存Markdown文件
            markdown_path = os.path.join(out_dir, os.path.basename(os.path.splitext(pdf_path)[0] + '.md'))
            try:
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_text)
                print(f"已生成Markdown文件: {markdown_path}")
            except Exception as e:
                print(f"保存Markdown文件失败: {e}")
            
            # 保存向量数据到CSV文件
            csv_path = os.path.join(out_dir, os.path.basename(os.path.splitext(pdf_path)[0] + '_vector.csv'))
            try:
                np.savetxt(csv_path, embedding, delimiter=',')
                print(f"已保存向量数据到: {csv_path}")
            except Exception as e:
                print(f"保存CSV文件失败: {e}")
            
            # 保存可视化图像
            if visualize and visualization_images:
                print(f"\n=== 可视化结果 ===")
                print(f"共有 {len(visualization_images)} 张可视化图像")
                save_visualization_images(visualization_images, viz_dir)
        
        except Exception as e:
            print(f"本地处理pipeline失败: {e}")
            print("尝试使用API方式处理PDF...")
            
            # 使用API处理PDF
            embedding, markdown_text, visualization_images = process_pdf_api(pdf_path, visualize)
            
            # 输出处理结果
            print("\n=== PDF处理结果（API方式）===")
            print(f"文件: {pdf_path}")
            print(f"向量维度: {embedding.shape}")
            print(f"向量前5个值: {embedding[:5]}")
            
            # 保存Markdown文件
            markdown_path = os.path.join(out_dir, os.path.basename(os.path.splitext(pdf_path)[0] + '.md'))
            try:
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_text)
                print(f"已生成Markdown文件: {markdown_path}")
            except Exception as e:
                print(f"保存Markdown文件失败: {e}")
            
            # 保存向量数据到CSV文件
            csv_path = os.path.join(out_dir, os.path.basename(os.path.splitext(pdf_path)[0] + '_vector.csv'))
            try:
                np.savetxt(csv_path, embedding, delimiter=',')
                print(f"已保存向量数据到: {csv_path}")
            except Exception as e:
                print(f"保存CSV文件失败: {e}")
            
            # 保存可视化图像
            if visualize and visualization_images:
                print(f"\n=== 可视化结果 ===")
                print(f"共有 {len(visualization_images)} 张可视化图像")
                save_visualization_images(visualization_images, viz_dir)
    
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()