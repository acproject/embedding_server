import os
import numpy as np
from PIL import Image
import logging
from typing import Optional, Dict, Any, List, Union

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
    
    def extract_formula(self, image: Union[np.ndarray, Image.Image]) -> str:
        """
        从图像中提取公式
        
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