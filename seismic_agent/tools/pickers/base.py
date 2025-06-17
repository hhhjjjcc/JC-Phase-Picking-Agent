"""
基础震相拾取器类
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BasePicker(ABC):
    """震相拾取器基类"""
    
    def __init__(self):
        """初始化拾取器"""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def pick(self, st, **kwargs) -> Dict[str, Any]:
        """
        执行震相拾取
        
        Args:
            st: ObsPy Stream对象
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 拾取结果
        """
        pass
    
    def __del__(self):
        """清理资源"""
        pass 