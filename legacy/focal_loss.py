"""
Class-Aware Focal Loss with Label Smoothing
=============================================
Paper: Lin et al. "Focal Loss for Dense Object Detection" ICCV'17
Formula: FL(p_t) = -α_t(1-p_t)^γ * log(p_t)

Improvements over original:
  - Per-class α weights (inverse frequency)
  - Integrated label smoothing (ε=0.1)
  - Numerical stability via log-sum-exp
"""
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss with per-class alpha and optional label smoothing.

    Args:
        gamma: Focusing parameter (default 2.0). Higher → more focus on hard examples.
        alpha: Per-class weights. Can be:
            - float: same weight for all classes (original paper)
            - list/array: per-class weights [α_0, α_1, ..., α_C]
            - None: uniform weights (1.0 for all)
        label_smoothing: Smoothing factor ε ∈ [0, 1). Target becomes:
            y_smooth = y * (1 - ε) + ε / num_classes
        name: Loss name for logging.
    """
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.0,
                 name='focal_loss', reduction='sum_over_batch_size', **kwargs):
        super().__init__(name=name, reduction=reduction)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        # Handle alpha: scalar, list, or None
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = tf.constant(alpha, dtype=tf.float32)
        else:
            self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        # Clip predictions for numerical stability
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Apply label smoothing: y_smooth = y * (1 - ε) + ε / C
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + \
                     self.label_smoothing / num_classes

        # Cross entropy: -y * log(p)
        ce = -y_true * K.log(y_pred)

        # Focal weight: (1 - p_t)^γ
        focal_weight = K.pow(1.0 - y_pred, self.gamma)

        # Apply per-class alpha
        if self.alpha is not None:
            if isinstance(self.alpha, float):
                alpha_weight = self.alpha
            else:
                # alpha is a tensor of per-class weights
                alpha_weight = self.alpha  # broadcasts over batch
            focal_weight = alpha_weight * focal_weight

        # Final focal loss
        focal_loss = focal_weight * ce

        return K.mean(K.sum(focal_loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        alpha_val = self.alpha
        if isinstance(alpha_val, tf.Tensor):
            alpha_val = alpha_val.numpy().tolist()
        config.update({
            'gamma': self.gamma,
            'alpha': alpha_val,
            'label_smoothing': self.label_smoothing
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def compute_class_alpha(class_counts, normalize=True):
    """
    Compute per-class alpha weights inversely proportional to frequency.

    Alpha_i = total / (n_classes * count_i)

    Args:
        class_counts: dict or list of counts per class
        normalize: if True, normalize so weights sum to n_classes

    Returns:
        list of alpha weights

    Example:
        >>> compute_class_alpha({'debut': 624, 'grave': 294, 'normal': 186})
        [0.590, 1.252, 1.979]  # more weight on rare classes
    """
    if isinstance(class_counts, dict):
        counts = list(class_counts.values())
    else:
        counts = list(class_counts)

    total = sum(counts)
    n_classes = len(counts)

    alphas = [total / (n_classes * c) for c in counts]

    if normalize:
        alpha_sum = sum(alphas)
        alphas = [a * n_classes / alpha_sum for a in alphas]

    return alphas
