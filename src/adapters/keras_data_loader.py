import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Callable, Optional
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.ports.data_port import DataPort

class KerasDataLoader(DataPort):
    def get_train_test_generators(self, data_dir: str, img_size: Tuple[int, int], batch_size: int, preprocessing_fn: Optional[Callable] = None) -> Tuple[Any, Any, Dict[int, float]]:
        
        train_kwargs = {
            'rotation_range': 25,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'shear_range': 0.2,
            'zoom_range': 0.2,
            'horizontal_flip': True,
            'fill_mode': 'nearest'
        }
        test_kwargs = {}
        
        if preprocessing_fn:
            train_kwargs['preprocessing_function'] = preprocessing_fn
            test_kwargs['preprocessing_function'] = preprocessing_fn
        else:
            # Fallback: simple rescale si pas de preprocessing fourni
            train_kwargs['rescale'] = 1./255
            test_kwargs['rescale'] = 1./255
        
        train_datagen = ImageDataGenerator(**train_kwargs)
        test_datagen = ImageDataGenerator(**test_kwargs)
        
        train_dir = Path(data_dir) / 'train'
        test_dir = Path(data_dir) / 'test'
        
        class_counts = {}
        for class_name in ['benign', 'malignant', 'normal']:
            class_dir = train_dir / class_name
            if class_dir.exists():
                class_counts[class_name] = len(list(class_dir.glob('*.png')))
        
        total = sum(class_counts.values())
        if total == 0:
            raise ValueError(f"No images found in {train_dir}. Please check your dataset path.")
            
        class_weights = {
            0: total / (3 * max(1, class_counts.get('benign', 1))),
            1: total / (3 * max(1, class_counts.get('malignant', 1))),
            2: total / (3 * max(1, class_counts.get('normal', 1)))
        }
        
        print(f"\n📊 Class weights:")
        print(f"   Benign:    {class_weights[0]:.3f}")
        print(f"   Malignant: {class_weights[1]:.3f}")
        print(f"   Normal:    {class_weights[2]:.3f}\n")
        
        train_gen = train_datagen.flow_from_directory(
            str(train_dir),
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        test_gen = test_datagen.flow_from_directory(
            str(test_dir),
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_gen, test_gen, class_weights
