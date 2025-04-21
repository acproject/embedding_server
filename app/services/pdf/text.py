import os
import numpy as np
from PIL import Image
import logging
import cv2
from typing import Optional, Dict, Any, List, Tuple
import re

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
        self.ocr_model_cn = None  # 中文OCR模型
        self.ocr_model_en = None  # 英文OCR模型
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
            
            # 初始化中文模型
            logger.info("初始化中文OCR模型")
            self.ocr_model_cn = PaddleOCR(use_angle_cls=False, use_gpu=use_gpu, lang="ch", show_log=False)
            
            # 初始化英文模型
            logger.info("初始化英文OCR模型")
            # 检查是否存在英文模型
            en_model_dir = os.path.join(self.models_dir, "en_PP-OCRv4_mobile")
            if os.path.exists(en_model_dir):
                logger.info(f"使用自定义英文OCR模型: {en_model_dir}")
                try:
                    self.ocr_model_en = PaddleOCR(
                        use_angle_cls=False, 
                        use_gpu=use_gpu, 
                        lang="en",
                        rec_model_dir=en_model_dir,
                        show_log=False
                    )
                except Exception as e:
                    logger.warning(f"加载自定义英文模型失败: {e}，将使用默认英文模型")
                    self.ocr_model_en = PaddleOCR(use_angle_cls=False, use_gpu=use_gpu, lang="en", show_log=False)
            else:
                logger.info("使用默认英文OCR模型")
                self.ocr_model_en = PaddleOCR(use_angle_cls=False, use_gpu=use_gpu, lang="en", show_log=False)
            
            # 测试模型是否能正常工作
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)  # 创建一个空白测试图像
            try:
                _ = self.ocr_model_cn.ocr(test_image, cls=False)
                logger.info("中文OCR模型测试成功")
            except Exception as e:
                logger.warning(f"中文OCR模型测试失败: {e}")
                
            try:
                _ = self.ocr_model_en.ocr(test_image, cls=False)
                logger.info("英文OCR模型测试成功")
            except Exception as e:
                logger.warning(f"英文OCR模型测试失败: {e}")
            
            self.ocr_available = True
            logger.info("OCR模型加载成功")
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
    
    def _detect_language(self, image: np.ndarray) -> str:
        """
        检测图像中的主要语言
        
        Args:
            image: 图像，numpy数组格式
            
        Returns:
            str: 语言类型，"cn"表示中文，"en"表示英文
        """
        # 先尝试使用中文模型识别
        try:
            if self.ocr_model_cn is not None:
                result = self.ocr_model_cn.ocr(image, cls=False)
                if result and len(result) > 0:
                    texts = []
                    if isinstance(result[0], list):
                        for line in result[0]:
                            if isinstance(line, list) and len(line) >= 2:
                                text, _ = line[1]
                                texts.append(text)
                    
                    if texts:
                        # 统计中英文字符数量
                        cn_count = 0
                        en_count = 0
                        for text in texts:
                            # 统计中文字符
                            cn_count += len(re.findall(r'[\u4e00-\u9fff]', text))
                            # 统计英文字符
                            en_count += len(re.findall(r'[a-zA-Z]', text))
                        
                        # 根据字符数量判断语言类型
                        if cn_count > en_count:
                            logger.info(f"检测到中文文本，中文字符数: {cn_count}，英文字符数: {en_count}")
                            return "cn"
                        else:
                            logger.info(f"检测到英文文本，中文字符数: {cn_count}，英文字符数: {en_count}")
                            return "en"
        except Exception as e:
            logger.warning(f"语言检测失败: {e}")
        
        # 默认返回中文
        return "cn"
    
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
        
        # 检测语言类型
        lang = self._detect_language(image)
        logger.info(f"检测到的语言类型: {lang}")
        
        # 根据语言类型选择OCR模型
        ocr_model = self.ocr_model_cn if lang == "cn" else self.ocr_model_en
        
        # 使用选定的OCR模型提取文本
        if ocr_model is not None:
            try:
                result = ocr_model.ocr(image, cls=False)
                
                # 检查结果是否为None
                if result is None:
                    logger.warning(f"{lang}OCR模型返回None结果")
                    # 尝试使用另一个模型
                    alt_model = self.ocr_model_en if lang == "cn" else self.ocr_model_cn
                    if alt_model is not None:
                        logger.info(f"尝试使用{'英文' if lang == 'cn' else '中文'}OCR模型")
                        result = alt_model.ocr(image, cls=False)
                    
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
                        logger.info(f"{lang}OCR提取文本: {full_text[:100]}{'...' if len(full_text) > 100 else ''}")
                        return full_text
                    else:
                        logger.warning(f"{lang}OCR未提取到任何文本")
                else:
                    logger.warning(f"{lang}OCR返回空结果")
            except Exception as e:
                logger.error(f"{lang}OCR提取文本失败: {e}")
        
        # 如果PaddleOCR失败，尝试使用pytesseract
        try:
            import pytesseract
            # 根据语言类型选择pytesseract语言参数
            lang_param = 'chi_sim+eng' if lang == 'cn' else 'eng'
            text = pytesseract.image_to_string(Image.fromarray(image), lang=lang_param)
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
                # 根据语言类型选择pytesseract语言参数
                lang_param = 'chi_sim+eng' if lang == 'cn' else 'eng'
                text = pytesseract.image_to_string(Image.fromarray(binary), lang=lang_param)
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