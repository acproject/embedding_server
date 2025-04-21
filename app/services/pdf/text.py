import os
import numpy as np
from PIL import Image
import logging
import cv2  # 添加cv2导入
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class TextExtractor:
    """文本提取器"""
    
    def __init__(self, models_dir: str, device: str = "cpu"):
        """
        初始化文本提取器
        
        Args:
            models_dir: 模型目录
            device: 计算设备 ("cpu" 或 "cuda")
        """
        self.models_dir = models_dir
        self.device = device
        self.ocr_model = None
        self.ocr_available = False
        
        try:
            # 尝试初始化OCR模型
            self._init_ocr_model()
        except Exception as e:
            logger.error(f"OCR模型初始化失败: {e}")
            logger.warning("将使用备用方法进行文本提取")
    
    def _init_ocr_model(self):
        """初始化OCR模型"""
        # 优先尝试使用PaddleOCR
        try:
            from paddleocr import PaddleOCR
            import paddle
            
            # 设置Paddle内存分配策略，尝试解决内存问题
            paddle.device.set_device(self.device)
            if hasattr(paddle, 'set_flags'):
                paddle.set_flags({'FLAGS_allocator_strategy': 'auto_growth'})
            
            # 设置是否使用GPU
            use_gpu = self.device == "cuda"
            
            # 直接使用默认模型，避免自定义模型加载问题
            logger.info("TextExtractor使用默认PaddleOCR模型")
            self.ocr_model = PaddleOCR(use_angle_cls=False, use_gpu=use_gpu, lang="ch", show_log=False)
            
            # 测试模型是否能正常工作
            try:
                test_image = np.zeros((100, 100, 3), dtype=np.uint8)  # 创建一个空白测试图像
                _ = self.ocr_model.ocr(test_image, cls=False)
                logger.info("PaddleOCR模型测试成功")
            except Exception as e:
                logger.warning(f"PaddleOCR模型测试失败: {e}，将尝试重新初始化")
                # 尝试使用不同参数重新初始化
                self.ocr_model = PaddleOCR(use_angle_cls=False, use_gpu=use_gpu, lang="ch", 
                                           rec_char_dict_path=None, det=True, rec=True, show_log=False)
                
                self.ocr_available = True
                logger.info("PaddleOCR模型加载成功")
                return
        except ImportError as e:
            logger.warning(f"未安装PaddleOCR: {e}")
        except Exception as e:
            logger.error(f"PaddleOCR初始化失败: {e}")
        
        # 如果PaddleOCR失败，尝试使用pytesseract
        try:
            import pytesseract
            logger.info("使用pytesseract进行文本提取")
            self.ocr_available = True
        except ImportError:
            logger.warning("未安装pytesseract，文本提取功能可能受限")
            self.ocr_available = False
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        从图像中提取文本
        
        Args:
            image: 图像，numpy数组格式
            
        Returns:
            str: 提取的文本
        """
        if image is None or image.size == 0:
            logger.warning("输入图像为空")
            return ""
        
        # 确保图像是RGB格式
        if len(image.shape) == 2:
            # 灰度图转RGB
            image = np.stack([image] * 3, axis=2)
        elif image.shape[2] == 4:
            # RGBA转RGB
            image = image[:, :, :3]
        
        # 优先使用PaddleOCR
        if self.ocr_model is not None:
            try:
                result = self.ocr_model.ocr(image, cls=True)
                
                # 检查结果是否为None
                if result is None:
                    logger.warning("PaddleOCR返回None结果")
                    # 尝试不使用角度分类器
                    result = self.ocr_model.ocr(image, cls=False)
                    
                # 检查结果格式
                if result is not None and len(result) > 0:
                    texts = []
                    # 处理不同版本PaddleOCR返回的结果格式
                    if isinstance(result[0], list):
                        # 新版本PaddleOCR
                        for line in result[0]:
                            if isinstance(line, list) and len(line) >= 2:
                                text, confidence = line[1]
                                texts.append(text)
                    else:
                        # 旧版本PaddleOCR
                        for line in result:
                            if isinstance(line, dict) and 'text' in line:
                                texts.append(line['text'])
                            elif isinstance(line, list) and len(line) >= 2:
                                text = line[1][0] if isinstance(line[1], list) else line[1]
                                texts.append(text)
                    
                    if texts:
                        full_text = "\n".join(texts)
                        logger.info(f"PaddleOCR提取文本: {full_text[:100]}{'...' if len(full_text) > 100 else ''}")
                        return full_text
                    else:
                        logger.warning("PaddleOCR未提取到任何文本")
                else:
                    logger.warning("PaddleOCR返回空结果")
            except Exception as e:
                logger.error(f"PaddleOCR提取文本失败: {e}")
                # 尝试重新初始化OCR模型
                try:
                    from paddleocr import PaddleOCR
                    logger.info("尝试重新初始化PaddleOCR模型")
                    self.ocr_model = PaddleOCR(use_angle_cls=False, use_gpu=(self.device == "cuda"), lang="ch", show_log=False)
                    # 再次尝试提取
                    result = self.ocr_model.ocr(image, cls=False)
                    if result and len(result) > 0:
                        texts = []
                        for line in result[0]:
                            if len(line) >= 2:
                                text, confidence = line[1]
                                texts.append(text)
                            
                        full_text = "\n".join(texts)
                        logger.info(f"重新初始化后PaddleOCR提取文本: {full_text[:100]}{'...' if len(full_text) > 100 else ''}")
                        return full_text
                except Exception as e2:
                    logger.error(f"重新初始化PaddleOCR失败: {e2}")
        
        # 如果PaddleOCR失败，尝试使用pytesseract
        try:
            import pytesseract
            text = pytesseract.image_to_string(Image.fromarray(image), lang='chi_sim+eng')
            if text:
                logger.info(f"pytesseract提取文本: {text[:100]}{'...' if len(text) > 100 else ''}")
                return text
        except ImportError:
            logger.warning("未安装pytesseract，无法使用备用OCR")
        except Exception as e:
            logger.error(f"pytesseract提取文本失败: {e}")
        
        # 如果所有方法都失败，尝试使用简单的图像处理方法
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2).astype(np.uint8)
            else:
                gray = image.astype(np.uint8)
            
            # 二值化
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 尝试使用pytesseract再次识别二值化后的图像
            try:
                import pytesseract
                text = pytesseract.image_to_string(Image.fromarray(binary), lang='chi_sim+eng')
                if text:
                    logger.info(f"二值化后pytesseract提取文本: {text[:100]}{'...' if len(text) > 100 else ''}")
                    return text
            except:
                pass
            
            logger.warning("所有文本提取方法均失败")
            return ""
        except Exception as e:
            logger.error(f"简单图像处理方法失败: {e}")
            logger.warning("所有文本提取方法均失败")
            return ""