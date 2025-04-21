import os
import numpy as np
from PIL import Image
import logging
from typing import Optional, Dict, Any, List, Union
import base64
import io

logger = logging.getLogger(__name__)

class FormulaExtractor:
    """公式提取器"""
    
    def __init__(self, models_dir: str, device: str = "cpu"):
        """
        初始化公式提取器
        
        Args:
            models_dir: 模型目录
            device: 计算设备 ("cpu" 或 "cuda")
        """
        self.models_dir = models_dir
        self.device = device
        self.formula_model = None
        self.formula_available = False
        
        try:
            # 尝试初始化公式识别模型
            self._init_formula_model()
        except Exception as e:
            logger.error(f"公式识别模型初始化失败: {e}")
            logger.warning("公式识别功能将不可用")
    
    def _init_formula_model(self):
        """初始化公式识别模型"""
        try:
            # 尝试导入PaddleOCR
            from paddleocr import PaddleOCR
            import paddle
            
            # 设置Paddle内存分配策略，尝试解决内存问题
            paddle.device.set_device(self.device)
            if hasattr(paddle, 'set_flags'):
                paddle.set_flags({'FLAGS_allocator_strategy': 'auto_growth'})
            
            # 设置是否使用GPU
            use_gpu = self.device == "cuda"
            
            # 尝试直接使用默认模型，避免加载自定义模型时的错误
            logger.info("尝试使用默认模型进行公式识别")
            try:
                self.formula_model = PaddleOCR(use_angle_cls=False, use_gpu=use_gpu, lang="en", show_log=False)
                
                # 测试模型是否能正常工作
                test_image = np.zeros((100, 100, 3), dtype=np.uint8)  # 创建一个空白测试图像
                _ = self.formula_model.ocr(test_image, cls=False)
                
                self.formula_available = True
                logger.info("默认公式识别模型加载成功")
            except Exception as e:
                logger.error(f"默认公式模型加载失败: {e}")
                self.formula_available = False
                return
            
        except ImportError as e:
            logger.warning(f"未安装PaddleOCR，公式识别功能将不可用: {e}")
            self.formula_available = False
        except Exception as e:
            logger.error(f"公式识别模型初始化失败: {e}")
            self.formula_available = False
    
    def _image_to_base64(self, image: Union[np.ndarray, Image.Image]) -> str:
        """
        将图像转换为base64编码
        
        Args:
            image: 图像，numpy数组或PIL图像
            
        Returns:
            str: base64编码的图像
        """
        # 确保图像是PIL图像
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 将图像转换为base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def _crop_formula_area(self, image: Union[np.ndarray, Image.Image], box) -> Image.Image:
        """
        裁剪公式区域
        
        Args:
            image: 原始图像
            box: 公式区域坐标 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            
        Returns:
            Image.Image: 裁剪后的公式区域图像
        """
        # 确保图像是PIL图像
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 计算边界框
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        x0, y0 = max(0, int(min(x_coords))), max(0, int(min(y_coords)))
        x1, y1 = min(image.width, int(max(x_coords))), min(image.height, int(max(y_coords)))
        
        # 裁剪图像
        return image.crop((x0, y0, x1, y1))
    
    def recognize_formula(self, image: Union[np.ndarray, Image.Image]) -> str:
        """
        从图像中识别公式，如果识别区域小于图像的85%，则返回base64编码的图像
        
        Args:
            image: 图像，numpy数组或PIL图像
            
        Returns:
            str: 识别的公式文本或base64编码的图像
        """
        if not self.formula_available or self.formula_model is None:
            logger.warning("公式识别模型未初始化或不可用")
            # 如果模型不可用，直接返回图像的base64编码
            return self._image_to_base64(image)
        
        try:
            # 确保图像是numpy数组
            if isinstance(image, Image.Image):
                pil_image = image
                image_np = np.array(image)
            else:
                pil_image = Image.fromarray(image)
                image_np = image
            
            # 确保图像是RGB格式
            if len(image_np.shape) == 2:
                # 灰度图转RGB
                image_np = np.stack([image_np] * 3, axis=2)
            elif image_np.shape[2] == 4:
                # RGBA转RGB
                image_np = image_np[:, :, :3]
            
            # 获取图像尺寸
            img_height, img_width = image_np.shape[:2]
            img_area = img_height * img_width
            
            # 使用公式模型识别文本
            result = self.formula_model.ocr(image_np, cls=False)
            
            # 解析结果
            if result and len(result) > 0 and len(result[0]) > 0:
                # 计算所有识别区域的总面积
                total_formula_area = 0
                formula_boxes = []
                formula_texts = []
                
                for line in result[0]:
                    if len(line) >= 2:
                        box = line[0]  # 文本框坐标
                        text, confidence = line[1]  # 文本内容和置信度
                        
                        # 计算边界框
                        x_coords = [point[0] for point in box]
                        y_coords = [point[1] for point in box]
                        x0, y0 = min(x_coords), min(y_coords)
                        x1, y1 = max(x_coords), max(y_coords)
                        
                        # 计算区域面积
                        area = (x1 - x0) * (y1 - y0)
                        total_formula_area += area
                        
                        formula_boxes.append(box)
                        formula_texts.append(text)
                
                # 计算识别区域占总图像的比例
                area_ratio = total_formula_area / img_area
                logger.debug(f"公式区域占比: {area_ratio:.2f}")
                
                # 如果识别区域小于图像的85%，返回base64编码的图像
                if area_ratio < 0.85:
                    logger.info(f"公式区域占比小于85%，返回base64编码的图像")
                    
                    # 如果只有一个公式区域，裁剪该区域
                    if len(formula_boxes) == 1:
                        cropped_img = self._crop_formula_area(pil_image, formula_boxes[0])
                        return self._image_to_base64(cropped_img)
                    else:
                        # 多个公式区域，返回整个图像
                        return self._image_to_base64(pil_image)
                else:
                    # 合并所有文本
                    full_text = " ".join(formula_texts)
                    logger.info(f"公式识别结果: {full_text[:100]}{'...' if len(full_text) > 100 else ''}")
                    return full_text
            else:
                logger.warning("未识别到公式，返回base64编码的图像")
                return self._image_to_base64(pil_image)
            
        except Exception as e:
            logger.error(f"公式识别失败: {e}")
            # 出错时返回base64编码的图像
            if 'pil_image' in locals():
                return self._image_to_base64(pil_image)
            else:
                return self._image_to_base64(image)
    
    def extract_formula(self, image: Union[np.ndarray, Image.Image]) -> str:
        """
        从图像中提取公式（兼容旧接口）
        
        Args:
            image: 图像，numpy数组或PIL图像
            
        Returns:
            str: 提取的公式文本
        """
        if not self.formula_available or self.formula_model is None:
            logger.warning("公式识别模型未初始化或不可用")
            return ""
        
        try:
            # 确保图像是numpy数组
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # 确保图像是RGB格式
            if len(image.shape) == 2:
                # 灰度图转RGB
                image = np.stack([image] * 3, axis=2)
            elif image.shape[2] == 4:
                # RGBA转RGB
                image = image[:, :, :3]
            
            # 使用公式模型识别文本
            result = self.formula_model.ocr(image, cls=False)
            
            # 解析结果
            if result and len(result) > 0:
                texts = []
                for line in result[0]:
                    if len(line) >= 2:
                        text, confidence = line[1]
                        texts.append(text)
                
                # 合并所有文本
                full_text = " ".join(texts)
                logger.info(f"公式识别结果: {full_text[:100]}{'...' if len(full_text) > 100 else ''}")
                return full_text
            else:
                logger.warning("未识别到公式")
                return ""
            
        except Exception as e:
            logger.error(f"公式识别失败: {e}")
            return ""
    
    def test_formula_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        测试从图像中提取公式的功能
        
        Args:
            image_path: 图像路径
            
        Returns:
            Dict: 包含检测和识别结果的字典
        """
        if not self.formula_available or self.formula_model is None:
            logger.warning("公式识别模型未初始化或不可用")
            return {
                "detection": {"count": 0, "boxes": []},
                "recognition": {"formulas": []}
            }
        
        try:
            # 加载图像
            if not os.path.exists(image_path):
                logger.error(f"图像文件不存在: {image_path}")
                return {
                    "detection": {"count": 0, "boxes": []},
                    "recognition": {"formulas": []}
                }
            
            image = Image.open(image_path)
            image_np = np.array(image)
            
            # 确保图像是RGB格式
            if len(image_np.shape) == 2:
                # 灰度图转RGB
                image_np = np.stack([image_np] * 3, axis=2)
            elif image_np.shape[2] == 4:
                # RGBA转RGB
                image_np = image_np[:, :, :3]
            
            # 使用公式模型识别文本
            result = self.formula_model.ocr(image_np, cls=False)
            
            # 解析结果
            formulas = []
            boxes = []
            
            if result and len(result) > 0:
                for line in result[0]:
                    if len(line) >= 2:
                        box = line[0]  # 文本框坐标，格式为[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                        text, confidence = line[1]  # 文本内容和置信度
                        
                        # 计算边界框
                        x_coords = [point[0] for point in box]
                        y_coords = [point[1] for point in box]
                        x0, y0 = min(x_coords), min(y_coords)
                        x1, y1 = max(x_coords), max(y_coords)
                        
                        formulas.append({
                            "text": text,
                            "confidence": float(confidence),
                            "box": box,
                            "bbox": [float(x0), float(y0), float(x1), float(y1)]
                        })
                        
                        boxes.append({
                            "box": box,
                            "bbox": [float(x0), float(y0), float(x1), float(y1)]
                        })
            
            logger.info(f"公式识别结果: 检测到 {len(formulas)} 个公式")
            
            return {
                "detection": {"count": len(boxes), "boxes": boxes},
                "recognition": {"formulas": formulas}
            }
            
        except Exception as e:
            logger.error(f"公式测试失败: {e}")
            return {
                "detection": {"count": 0, "boxes": []},
                "recognition": {"formulas": []}
            }