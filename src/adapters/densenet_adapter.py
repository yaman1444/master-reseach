from typing import Tuple, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121

from src.ports.model_port import ModelPort
from src.domain.cbam import CBAM

class DenseNetAdapter(ModelPort):
    def build_model(self, img_size: Tuple[int, int], dropout_rate: float, l2_reg: float) -> Tuple[Any, Any]:
        base_model = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=(*img_size, 3)
        )
        base_model.trainable = False
        
        inputs = keras.Input(shape=(*img_size, 3))
        x = base_model(inputs, training=False)
        x = CBAM(reduction_ratio=16)(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Dense(512, activation='relu', 
                         kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(256, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        outputs = layers.Dense(3, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        return model, base_model
