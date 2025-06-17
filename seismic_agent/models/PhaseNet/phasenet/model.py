"""
PhaseNet模型定义
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import logging

logger = logging.getLogger(__name__)

class ModelConfig:
    """模型配置类"""
    
    def __init__(self,
                 X_shape,
                 Y_shape,
                 batch_size=1,
                 depths=[5, 5, 5, 5, 5],
                 filters_root=8,
                 kernel_size=7,
                 pool_size=4,
                 dilation_rate=1,
                 class_weights=[1, 1, 1],
                 loss_type='cross_entropy',
                 optimizer='adam',
                 learning_rate=0.001):
        """
        初始化模型配置
        
        Args:
            X_shape: 输入形状
            Y_shape: 输出形状
            batch_size: 批处理大小
            depths: 每层的深度
            filters_root: 根过滤器数量
            kernel_size: 卷积核大小
            pool_size: 池化大小
            dilation_rate: 扩张率
            class_weights: 类别权重
            loss_type: 损失函数类型
            optimizer: 优化器
            learning_rate: 学习率
        """
        self.X_shape = X_shape
        self.Y_shape = Y_shape
        self.batch_size = batch_size
        self.depths = depths
        self.filters_root = filters_root
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dilation_rate = dilation_rate
        self.class_weights = class_weights
        self.loss_type = loss_type
        self.optimizer = optimizer
        self.learning_rate = learning_rate

def crop_and_concat(x1, x2):
    """
    裁剪并连接两个张量
    
    Args:
        x1: 第一个张量
        x2: 第二个张量
        
    Returns:
        tensor: 连接后的张量
    """
    # 确保x1和x2具有相同的形状
    x1_shape = x1.get_shape().as_list()
    x2_shape = x2.get_shape().as_list()
    
    # 计算偏移量
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    
    # 裁剪x1
    x1_crop = tf.slice(x1, offsets, size)
    
    # 连接
    return tf.concat([x1_crop, x2], 3)

def crop_only(x1, x2):
  """
    裁剪张量以匹配目标形状
    
    Args:
        x1: 要裁剪的张量
        x2: 目标张量
        
    Returns:
        tensor: 裁剪后的张量
  """
    x1_shape = x1.get_shape().as_list()
    x2_shape = x2.get_shape().as_list()
    
    # 计算偏移量
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    
    # 裁剪
    return tf.slice(x1, offsets, size)

class UNet(Model):
    """U-Net模型"""
    
    def __init__(self, config):
        """
        初始化U-Net模型
        
        Args:
            config: 模型配置
        """
        super(UNet, self).__init__()
        self.config = config
        self._build_model()
        
    def _build_model(self):
        """构建模型"""
        # 输入层
        self.input_layer = layers.Input(shape=self.config.X_shape)
        
        # 编码器路径
        x = self.input_layer
        
        # 第一层卷积
        x = layers.Conv2D(self.config.filters_root, self.config.kernel_size, padding='same', activation='relu')(x)
        
        # 下采样路径
        for i in range(len(self.config.depths)):
            # 池化层
            x = layers.MaxPooling2D(pool_size=(1, self.config.pool_size), padding='same')(x)
            
            # 卷积层
            filters = self.config.filters_root * (2 ** i)
            x = layers.Conv2D(filters, self.config.kernel_size, padding='same', activation='relu')(x)
            x = layers.Conv2D(filters, self.config.kernel_size, padding='same', activation='relu')(x)
            
        # 解码器路径
        for i in range(len(self.config.depths)-1, -1, -1):
            # 上采样
            x = layers.UpSampling2D(size=(1, self.config.pool_size))(x)
            
            # 卷积层
            filters = self.config.filters_root * (2 ** i)
            x = layers.Conv2D(filters, self.config.kernel_size, padding='same', activation='relu')(x)
            x = layers.Conv2D(filters, self.config.kernel_size, padding='same', activation='relu')(x)
            
        # 输出层
        self.output_layer = layers.Conv2D(self.config.Y_shape[-1], 1, padding='same', activation='softmax')(x)
        
        # 构建模型
        self.model = Model(inputs=self.input_layer, outputs=self.output_layer)
        
        # 编译模型
        self.model.compile(
            optimizer=self.config.optimizer,
            loss=self._get_loss(),
            metrics=['accuracy']
        )

    def _get_loss(self):
        """获取损失函数"""
        if self.config.loss_type == 'cross_entropy':
            return 'categorical_crossentropy'
        elif self.config.loss_type == 'focal':
            return self._focal_loss
        else:
            raise ValueError(f"不支持的损失函数类型: {self.config.loss_type}")
            
    def _focal_loss(self, y_true, y_pred):
        """Focal Loss"""
        gamma = 2.0
        alpha = 0.25
        
        # 计算交叉熵
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # 计算权重
        weights = tf.reduce_sum(y_true * self.config.class_weights, axis=-1)
        
        # 计算focal loss
        focal_loss = alpha * weights * tf.pow(1 - y_pred, gamma) * cross_entropy
        
        return tf.reduce_mean(focal_loss)
        
    def call(self, inputs):
        """前向传播"""
        return self.model(inputs)
        
    def predict(self, x):
        """预测"""
        return self.model.predict(x)
        
    def save_weights(self, filepath):
        """保存权重"""
        self.model.save_weights(filepath)
        
    def load_weights(self, filepath):
        """加载权重"""
        self.model.load_weights(filepath)
