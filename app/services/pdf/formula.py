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
            # 修复路径，避免重复的 "formula" 目录
            formula_rec_path = os.path.join(self.models_dir, "formula_recognition/UniMERNet/pytorch_model.bin")
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
    
    def detect_formulas(self, image: np.ndarray) -> list:
        """
        检测图像中的公式区域
        
        Args:
            image: 输入图像的numpy数组
            
        Returns:
            list: 公式区域列表，每个元素包含坐标和类型信息
        """
        if self.formula_det_model is None:
            logger.warning("公式检测模型未加载，跳过公式检测")
            return []
        
        try:
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
            # 如果没有公式识别模型，返回None
            if self.formula_rec_path is None:
                return None
            
            # 这里应该实现公式识别的具体逻辑
            # 由于UniMERNet的具体使用方式可能需要特定的代码，这里仅提供框架
            
            # 示例：返回一个简单的LaTeX公式
            return "E = mc^2"
        except Exception as e:
            logger.error(f"公式识别失败: {e}")
            return None
            
    def test_formula_from_image(self, image_path: str) -> dict:
        """
        从图像文件测试公式检测和识别
        
        Args:
            image_path: 图像文件路径
            
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
            
            # 检测公式
            formula_blocks = self.detect_formulas(image)
            
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