"""
CBAM: Convolutional Block Attention Module
Paper: Woo et al. "CBAM: Convolutional Block Attention Module" ECCV'18
Sequential channel + spatial attention for feature refinement
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Reshape, Multiply, Activation, Concatenate

class ChannelAttention(Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
    
    def build(self, input_shape):
        channels = input_shape[-1]
        self.shared_dense1 = Dense(channels // self.ratio, activation='relu')
        self.shared_dense2 = Dense(channels, activation='sigmoid')
        super().build(input_shape)
    
    def call(self, inputs):
        # Global pooling
        avg_pool = GlobalAveragePooling2D()(inputs)
        max_pool = GlobalMaxPooling2D()(inputs)
        
        # Shared MLP
        avg_out = self.shared_dense2(self.shared_dense1(avg_pool))
        max_out = self.shared_dense2(self.shared_dense1(max_pool))
        
        # Channel attention
        attention = avg_out + max_out
        attention = Reshape((1, 1, -1))(attention)
        
        return Multiply()([inputs, attention])

class SpatialAttention(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
    
    def build(self, input_shape):
        self.conv = Conv2D(1, self.kernel_size, padding='same', activation='sigmoid')
        super().build(input_shape)
    
    def call(self, inputs):
        # Channel pooling
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        
        # Concatenate and convolve
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        attention = self.conv(concat)
        
        return Multiply()([inputs, attention])

class CBAM(Layer):
    """Complete CBAM module: Channel → Spatial attention"""
    def __init__(self, reduction_ratio=8, ratio=None, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        # Support both 'ratio' (old) and 'reduction_ratio' (new)
        if ratio is not None:
            self.reduction_ratio = ratio
        else:
            self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.channel_attention = ChannelAttention(self.reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
            'kernel_size': self.kernel_size
        })
        return config
    
    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x
