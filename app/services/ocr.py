import os
import numpy as np
from PIL import Image
from loguru import logger
from typing import Optional, List, Dict, Any, Union

class OCRService:
    """OCR服务，用于文本识别"""
    
    def __init__(self, models_dir: str, device: str = "cpu"):
        """
        初始化OCR服务
        
        Args:
            models_dir: 模型目录
            device: 计算设备 ("cpu" 或 "cuda")
        """
        self.models_dir = models_dir
        self.device = device
        self.ocr_model = None
        
        try:
            self._init_ocr_model()
        except Exception as e:
            logger.error(f"OCR模型初始化失败: {e}")
            raise
    
    def _init_ocr_model(self):
        """初始化OCR模型"""
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
            
            # OCR模型路径
            ocr_model_dir = os.path.join(self.models_dir, "ocr")
            
            # 检查模型文件是否存在
            if not os.path.exists(ocr_model_dir):
                logger.warning(f"OCR模型目录不存在: {ocr_model_dir}，将使用默认模型")
                # 使用默认模型
                self.ocr_model = PaddleOCR(use_angle_cls=False, use_gpu=use_gpu, lang="ch", show_log=False)
            else:
                # 检查是否是v4版本模型（单一inference.pdiparams文件）
                inference_model = os.path.join(ocr_model_dir, "inference.pdiparams")
                if os.path.exists(inference_model):
                    # 检查模型文件大小，确保文件完整
                    model_size = os.path.getsize(inference_model)
                    logger.info(f"v4模型文件大小: {model_size} 字节")
                    
                    if model_size < 1000:  # 如果文件太小，可能是损坏的
                        logger.warning(f"v4模型文件可能损坏，大小仅为 {model_size} 字节")
                        raise ValueError(f"模型文件可能损坏: {inference_model}")
                    
                    # 使用v4版本模型
                    logger.info(f"使用v4版本OCR模型: {ocr_model_dir}")
                    
                    # 尝试使用不同的参数组合
                    try:
                        self.ocr_model = PaddleOCR(
                            use_angle_cls=False,  # 禁用角度分类器，减少内存使用
                            use_gpu=use_gpu,
                            lang="ch",
                            rec_model_dir=ocr_model_dir,  # 直接使用主目录
                            show_log=False
                        )
                    except Exception as e1:
                        logger.warning(f"加载v4模型第一次尝试失败: {e1}")
                        # 第二次尝试，使用不同的参数
                        try:
                            self.ocr_model = PaddleOCR(
                                use_angle_cls=False,
                                use_gpu=use_gpu,
                                lang="ch",
                                det=False,  # 禁用检测模型
                                rec=True,   # 只使用识别模型
                                rec_model_dir=ocr_model_dir,
                                show_log=False
                            )
                        except Exception as e2:
                            logger.warning(f"加载v4模型第二次尝试失败: {e2}")
                            # 第三次尝试，使用默认模型
                            self.ocr_model = PaddleOCR(use_angle_cls=False, use_gpu=use_gpu, lang="ch", show_log=False)
                            logger.info("回退到使用默认PaddleOCR模型")
                else:
                    # 检查是否存在传统的det/rec/cls目录结构
                    if (os.path.exists(os.path.join(ocr_model_dir, "det")) or 
                        os.path.exists(os.path.join(ocr_model_dir, "rec")) or 
                        os.path.exists(os.path.join(ocr_model_dir, "cls"))):
                        # 使用传统模型结构
                        logger.info(f"使用传统结构OCR模型: {ocr_model_dir}")
                        self.ocr_model = PaddleOCR(
                            use_angle_cls=True,
                            use_gpu=use_gpu,
                            lang="ch",
                            det_model_dir=os.path.join(ocr_model_dir, "det") if os.path.exists(os.path.join(ocr_model_dir, "det")) else None,
                            rec_model_dir=os.path.join(ocr_model_dir, "rec") if os.path.exists(os.path.join(ocr_model_dir, "rec")) else None,
                            cls_model_dir=os.path.join(ocr_model_dir, "cls") if os.path.exists(os.path.join(ocr_model_dir, "cls")) else None,
                            show_log=False
                        )
                    else:
                        # 找不到任何有效的模型文件，使用默认模型
                        logger.warning(f"在 {ocr_model_dir} 中找不到有效的OCR模型文件，将使用默认模型")
                        self.ocr_model = PaddleOCR(use_angle_cls=False, use_gpu=use_gpu, lang="ch", show_log=False)
            
            # 测试模型是否能正常工作
            try:
                test_image = np.zeros((100, 100, 3), dtype=np.uint8)  # 创建一个空白测试图像
                _ = self.ocr_model.ocr(test_image, cls=False)
                logger.info("OCR模型测试成功")
            except Exception as e:
                logger.error(f"OCR模型测试失败: {e}")
                # 尝试使用默认模型作为备选
                logger.info("尝试使用默认OCR模型")
                self.ocr_model = PaddleOCR(use_angle_cls=False, use_gpu=use_gpu, lang="ch", show_log=False)
                # 再次测试
                test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                _ = self.ocr_model.ocr(test_image, cls=False)
            
            logger.info("OCR模型加载成功")
            
        except ImportError as e:
            logger.error(f"未安装PaddleOCR: {e}")
            raise
        except Exception as e:
            logger.error(f"OCR模型初始化失败: {e}")
            raise
    
    def recognize_text(self, image: Union[np.ndarray, Image.Image]) -> str:
        """
        识别图像中的文本
        
        Args:
            image: 图像，numpy数组或PIL图像
            
        Returns:
            str: 识别出的文本
        """
        if self.ocr_model is None:
            logger.error("OCR模型未初始化")
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
            
            # 使用OCR模型识别文本
            result = self.ocr_model.ocr(image, cls=True)
            
            # 解析结果
            if result and len(result) > 0:
                texts = []
                for line in result[0]:
                    if len(line) >= 2:
                        text, confidence = line[1]
                        texts.append(text)
                
                # 合并所有文本
                full_text = "\n".join(texts)
                logger.info(f"OCR识别结果: {full_text[:100]}{'...' if len(full_text) > 100 else ''}")
                return full_text
            else:
                logger.warning("OCR未识别到文本")
                return ""
            
        except Exception as e:
            logger.error(f"OCR识别失败: {e}")
            return ""
    
    def recognize_text_with_layout(self, image: Union[np.ndarray, Image.Image]) -> List[Dict[str, Any]]:
        """
        识别图像中的文本，并保留布局信息
        
        Args:
            image: 图像，numpy数组或PIL图像
            
        Returns:
            List[Dict]: 文本块列表，每个块包含文本、坐标和置信度
        """
        if self.ocr_model is None:
            logger.error("OCR模型未初始化")
            return []
        
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
            
            # 使用OCR模型识别文本
            result = self.ocr_model.ocr(image, cls=True)
            
            # 解析结果
            text_blocks = []
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
                        
                        text_blocks.append({
                            "text": text,
                            "confidence": confidence,
                            "box": box,
                            "bbox": [x0, y0, x1, y1]
                        })
                
                logger.info(f"OCR识别到 {len(text_blocks)} 个文本块")
            else:
                logger.warning("OCR未识别到文本")
            
            return text_blocks
            
        except Exception as e:
            logger.error(f"OCR识别失败: {e}")
            return []