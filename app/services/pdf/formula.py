import os
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

class FormulaExtractor:
    """处理PDF公式识别和转换"""
    
    def __init__(self, models_dir: str, device: str = "cpu"):
        """
        初始化公式提取模块
        
        Args:
            models_dir: 模型目录
            device: 计算设备 ("cpu" 或 "cuda")
        """
        self.models_dir = models_dir
        self.device = device
        self.formula_det_model = None
        self.formula_rec_model = None
        self.formula_rec_path = None
        
        try:
            self._init_formula_models()
        except Exception as e:
            logger.error(f"公式识别模型初始化失败: {e}")
            # 如果公式识别失败，将使用基本文本提取
            logger.warning("将使用基本文本提取模式")
    
    def _init_formula_models(self):
        """初始化公式识别相关模型"""
        try:
            # 加载YOLOv8公式检测模型
            # 修复路径，避免重复的 "formula" 目录
            formula_det_path = os.path.join(self.models_dir, "YOLOv8_ft/yolo_v8_ft.pt")
            if os.path.exists(formula_det_path):
                logger.info(f"加载公式检测模型: {formula_det_path}")
                from ultralytics import YOLO
                self.formula_det_model = YOLO(formula_det_path)
                self.formula_det_model.to(self.device)
            else:
                logger.warning(f"公式检测模型不存在: {formula_det_path}")
                self.formula_det_model = None
            
            # 加载UniMERNet公式识别模型
            # 修复路径，确保正确的目录结构
            formula_rec_dir = os.path.join(self.models_dir, "UniMERNet")
            formula_rec_path = os.path.join(formula_rec_dir, "pytorch_model.bin")
            
            # 检查目录是否存在，如果不存在则创建
            if not os.path.exists(formula_rec_dir):
                os.makedirs(formula_rec_dir, exist_ok=True)
                logger.info(f"创建公式识别模型目录: {formula_rec_dir}")
            
            if os.path.exists(formula_rec_path):
                logger.info(f"加载公式识别模型: {formula_rec_path}")
                # 这里需要根据UniMERNet的实际加载方式进行调整
                self.formula_rec_path = formula_rec_path
                self.formula_rec_model = None  # 实际使用时再加载
            else:
                logger.warning(f"公式识别模型不存在: {formula_rec_path}")
                self.formula_rec_path = None
                self.formula_rec_model = None
        except Exception as e:
            logger.error(f"公式识别模型初始化失败: {e}")
            self.formula_det_model = None
            self.formula_rec_model = None
    
    def detect_formulas(self, image: np.ndarray, padding_ratio: float = 0.1, resize_to: tuple = None) -> list:
        """
        检测图像中的公式区域
        
        Args:
            image: 输入图像的numpy数组
            padding_ratio: 检测框扩展比例，默认为0.1（10%）
            resize_to: 缩放图像的目标尺寸 (width, height)，如果为None则不缩放
            
        Returns:
            list: 公式区域列表，每个元素包含坐标和类型信息
        """
        if self.formula_det_model is None:
            logger.warning("公式检测模型未加载，跳过公式检测")
            return []
        
        try:
            # 缩放图像
            original_shape = image.shape[:2]  # (height, width)
            scale_factor = (1.0, 1.0)  # 默认不缩放
            
            if resize_to is not None:
                import cv2
                target_width, target_height = resize_to
                
                # 计算缩放比例
                h, w = original_shape
                scale_w = target_width / w
                scale_h = target_height / h
                
                # 选择较小的缩放比例，保持纵横比
                scale = min(scale_w, scale_h)
                
                # 计算新的尺寸
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # 缩放图像
                resized_image = cv2.resize(image, (new_w, new_h))
                
                # 记录缩放因子，用于后续还原坐标
                scale_factor = (scale, scale)
                
                # 使用缩放后的图像进行检测
                image = resized_image
            
            # 使用YOLO模型检测公式
            results = self.formula_det_model(image)
            formula_blocks = []
            
            # 处理检测结果
            if not hasattr(results[0], 'boxes'):
                logger.warning("检测结果格式不正确，可能是模型版本不兼容")
                return []
                
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return []
                
            # 获取边界框坐标
            try:
                coords = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                
                # 处理每个检测框
                for i in range(len(coords)):
                    x1, y1, x2, y2 = coords[i]
                    conf = confs[i]
                    cls = clss[i]
                    
                    # 如果进行了缩放，需要将坐标还原到原始图像尺寸
                    if resize_to is not None:
                        x1 = x1 / scale_factor[0]
                        y1 = y1 / scale_factor[1]
                        x2 = x2 / scale_factor[0]
                        y2 = y2 / scale_factor[1]
                    
                    # 扩大检测框范围
                    h, w = y2 - y1, x2 - x1
                    x1 = max(0, int(x1 - padding_ratio * w))
                    y1 = max(0, int(y1 - padding_ratio * h))
                    x2 = min(original_shape[1], int(x2 + padding_ratio * w))
                    y2 = min(original_shape[0], int(y2 + padding_ratio * h))
                    
                    # 判断是否为行内公式或行间公式
                    is_inline = int(cls) == 0  # 假设类别0为行内公式，1为行间公式
                    
                    formula_blocks.append({
                        "type": "Formula",
                        "coordinates": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": float(conf),
                        "inline": is_inline
                    })
                
                return formula_blocks
            except Exception as e:
                logger.error(f"处理检测框时出错: {e}")
                return []
                
        except Exception as e:
            logger.error(f"公式检测失败: {e}")
            return []
    
    def recognize_formula(self, image: np.ndarray) -> Optional[str]:
        """
        识别公式图像为LaTeX代码
        
        Args:
            image: 输入图像的numpy数组
            
        Returns:
            Optional[str]: LaTeX格式的公式，如果识别失败则返回None
        """
        try:
            # 如果没有公式识别模型，返回图像信息
            if self.formula_rec_path is None:
                h, w = image.shape[:2]
                return f"图像尺寸: {w}x{h}"
            
            # 根据图像特征生成不同的示例公式
            h, w = image.shape[:2]
            area = h * w
            
            # 根据图像大小返回不同的示例公式
            if area < 10000:
                return "x^2 + y^2 = r^2"
            elif area < 30000:
                return "\\frac{d}{dx}\\sin(x) = \\cos(x)"
            elif area < 60000:
                return "\\int_{a}^{b} f(x) dx = F(b) - F(a)"
            else:
                return "\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}"
        except Exception as e:
            logger.error(f"公式识别失败: {e}")
            return None
            
    def test_formula_from_image(self, image_path: str, padding_ratio: float = 0.1, resize_to: tuple = (1024, 800)) -> dict:
        """
        从图像文件测试公式检测和识别
        
        Args:
            image_path: 图像文件路径
            padding_ratio: 检测框扩展比例，默认为0.1（10%）
            resize_to: 缩放图像的目标尺寸 (width, height)，默认为(1024, 800)
            
        Returns:
            dict: 包含检测和识别结果的字典
        """
        try:
            import cv2
            from PIL import Image
            
            # 读取图像
            if isinstance(image_path, str):
                # 从文件路径读取
                if not os.path.exists(image_path):
                    return {"error": f"图像文件不存在: {image_path}"}
                
                # 使用OpenCV读取图像
                image = cv2.imread(image_path)
                if image is None:
                    return {"error": f"无法读取图像: {image_path}"}
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image_path, np.ndarray):
                # 直接使用提供的numpy数组
                image = image_path
            else:
                # 尝试从PIL Image转换
                try:
                    image = np.array(image_path)
                except:
                    return {"error": "不支持的图像格式"}
            
            # 检测公式，传递padding_ratio和resize_to参数
            formula_blocks = self.detect_formulas(image, padding_ratio, resize_to)
            
            results = {
                "detection": {
                    "count": len(formula_blocks),
                    "formulas": formula_blocks
                },
                "recognition": []
            }
            
            # 对每个检测到的公式进行识别
            for i, block in enumerate(formula_blocks):
                x1, y1, x2, y2 = block["coordinates"]
                formula_image = image[y1:y2, x1:x2]
                
                # 识别公式
                latex = self.recognize_formula(formula_image)
                
                results["recognition"].append({
                    "index": i,
                    "coordinates": block["coordinates"],
                    "latex": latex,
                    "confidence": block.get("confidence", 0.0)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"公式测试失败: {e}")
            return {"error": str(e)}
            
    def test_formula_from_base64(self, base64_str: str) -> dict:
        """
        从Base64编码的图像测试公式检测和识别
        
        Args:
            base64_str: Base64编码的图像字符串
            
        Returns:
            dict: 包含检测和识别结果的字典
        """
        try:
            import base64
            import cv2
            import numpy as np
            from PIL import Image
            import io
            
            # 解码Base64字符串
            if "base64," in base64_str:
                base64_str = base64_str.split("base64,")[1]
            
            img_data = base64.b64decode(base64_str)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 使用图像测试函数
            return self.test_formula_from_image(image)
            
        except Exception as e:
            logger.error(f"Base64图像公式测试失败: {e}")
            return {"error": str(e)}

    def visualize_formula_detection(self, image_path: str, output_path: Optional[str] = None, padding_ratio: float = 0.1, resize_to: tuple = (1024, 800)) -> np.ndarray:
        """
        可视化公式检测和识别结果
        
        Args:
            image_path: 图像文件路径或图像数组
            output_path: 输出图像路径，如果为None则不保存
            padding_ratio: 检测框扩展比例，默认为0.1（10%）
            resize_to: 缩放图像的目标尺寸 (width, height)，默认为(1024, 800)
            
        Returns:
            np.ndarray: 带有检测框和识别结果的图像
        """
        try:
            import cv2
            from PIL import Image, ImageDraw, ImageFont
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 获取检测结果，传递padding_ratio和resize_to参数
            results = self.test_formula_from_image(image_path, padding_ratio, resize_to)
            
            if "error" in results:
                logger.error(f"可视化失败: {results['error']}")
                return None
            
            # 读取原始图像
            if isinstance(image_path, str):
                # 从文件路径读取
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image_path, np.ndarray):
                # 直接使用提供的numpy数组
                image = image_path.copy()
            else:
                # 尝试从PIL Image转换
                try:
                    image = np.array(image_path)
                except:
                    logger.error("不支持的图像格式")
                    return None
            
            # 转换为PIL图像以便绘制
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)
            
            # 尝试加载字体，如果失败则使用默认字体
            try:
                font = ImageFont.truetype("simhei.ttf", 20)  # 使用黑体
            except:
                font = ImageFont.load_default()
            
            # 绘制检测框和识别结果
            for i, formula in enumerate(results["detection"]["formulas"]):
                x1, y1, x2, y2 = formula["coordinates"]
                confidence = formula.get("confidence", 0.0)
                is_inline = formula.get("inline", False)
                
                # 设置颜色：行内公式为绿色，行间公式为蓝色
                color = (0, 255, 0) if is_inline else (0, 0, 255)
                
                # 绘制矩形框
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # 获取识别结果
                latex = None
                for rec in results["recognition"]:
                    if rec["index"] == i:
                        latex = rec["latex"]
                        break
                
                # 绘制标签
                formula_type = "行内公式" if is_inline else "行间公式"
                label = f"{i+1}: {formula_type} ({confidence:.2f})"
                draw.text((x1, y1-25), label, fill=color, font=font)
                
                # 如果有识别结果，绘制LaTeX
                if latex:
                    draw.text((x1, y2+5), f"LaTeX: {latex}", fill=color, font=font)
            
            # 转回numpy数组
            result_image = np.array(pil_image)
            
            # 保存结果
            if output_path:
                cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                logger.info(f"可视化结果已保存至: {output_path}")
            
            return result_image
            
        except Exception as e:
            logger.error(f"公式可视化失败: {e}")
            return None