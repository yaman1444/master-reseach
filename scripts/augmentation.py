"""
Advanced augmentation: CLAHE + Mixup/CutMix
Mixup: Zhang et al. "mixup: Beyond Empirical Risk Minimization" ICLR'18
Formula: x̃ = λx_i + (1-λ)x_j, ỹ = λy_i + (1-λ)y_j where λ~Beta(α,α)
"""
import numpy as np
import cv2
import tensorflow as tf

def apply_clahe(image):
    """Contrast Limited Adaptive Histogram Equalization for medical images"""
    image_uint8 = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        enhanced = clahe.apply(image_uint8)
    
    return enhanced.astype(np.float32) / 255.0

def mixup_batch(x, y, alpha=0.2):
    """
    Mixup augmentation: λ~Beta(α,α)
    Returns mixed inputs and labels
    """
    batch_size = x.shape[0]
    lam = np.random.beta(alpha, alpha, batch_size)
    
    # Reshape lambda for broadcasting
    lam_x = lam.reshape(batch_size, 1, 1, 1)
    lam_y = lam.reshape(batch_size, 1)
    
    # Random permutation
    indices = np.random.permutation(batch_size)
    
    # Mix inputs and labels
    mixed_x = lam_x * x + (1 - lam_x) * x[indices]
    mixed_y = lam_y * y + (1 - lam_y) * y[indices]
    
    return mixed_x, mixed_y

def cutmix_batch(x, y, alpha=0.2):
    """
    CutMix augmentation: Yun et al. "CutMix" ICCV'19
    Cut and paste patches with area ratio λ~Beta(α,α)
    """
    batch_size, h, w, c = x.shape
    lam = np.random.beta(alpha, alpha, batch_size)
    
    mixed_x = x.copy()
    mixed_y = y.copy()
    
    for i in range(batch_size):
        indices = np.random.permutation(batch_size)
        j = indices[0]
        
        # Random box coordinates with area ratio = 1-λ
        cut_ratio = np.sqrt(1.0 - lam[i])
        cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        x1 = np.clip(cx - cut_w // 2, 0, w)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cutmix
        mixed_x[i, y1:y2, x1:x2, :] = x[j, y1:y2, x1:x2, :]
        
        # Adjust lambda based on actual box area
        actual_lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        mixed_y[i] = actual_lam * y[i] + (1 - actual_lam) * y[j]
    
    return mixed_x, mixed_y

class AugmentedDataGenerator(tf.keras.utils.Sequence):
    """Custom generator with CLAHE + Mixup/CutMix"""
    def __init__(self, base_generator, use_clahe=True, use_mixup=True, 
                 mixup_alpha=0.2, mixup_prob=0.5):
        self.base_generator = base_generator
        self.use_clahe = use_clahe
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
    
    def __len__(self):
        return len(self.base_generator)
    
    def __getitem__(self, idx):
        x, y = self.base_generator[idx]
        
        # Apply CLAHE
        if self.use_clahe:
            for i in range(x.shape[0]):
                x[i] = apply_clahe(x[i])
        
        # Apply Mixup/CutMix with probability
        if self.use_mixup and np.random.rand() < self.mixup_prob:
            if np.random.rand() < 0.5:
                x, y = mixup_batch(x, y, self.mixup_alpha)
            else:
                x, y = cutmix_batch(x, y, self.mixup_alpha)
        
        return x, y
