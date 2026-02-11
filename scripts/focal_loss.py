"""
Focal Loss implementation for handling class imbalance
Paper: Lin et al. "Focal Loss for Dense Object Detection" ICCV'17
Formula: FL(p_t) = -α_t(1-p_t)^γ * log(p_t)
"""
import tensorflow as tf
from tensorflow.keras import backend as K

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name='focal_loss', reduction='sum_over_batch_size', **kwargs):
        super().__init__(name=name, reduction=reduction)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        ce = -y_true * K.log(y_pred)
        
        # Calculate focal loss: FL = -α(1-p_t)^γ * log(p_t)
        focal_weight = self.alpha * K.pow(1 - y_pred, self.gamma)
        focal_loss = focal_weight * ce
        
        return K.mean(K.sum(focal_loss, axis=-1))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
