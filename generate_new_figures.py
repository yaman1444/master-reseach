import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import sys

sys.path.append('/Users/yaman/master-reseach/src')
from adapters.keras_data_loader import KerasDataLoader
from presentation.config import CONFIG
from src.domain.preprocessing import apply_clahe
from src.domain.focal_loss import FocalLoss

from src.domain.cbam import CBAM

def evaluate_and_plot_cm(model_name, dataset_type='clean'):
    print(f"Generating confusion matrix for {model_name} on {dataset_type} dataset...")
    
    # Load dataset
    data_dir = CONFIG['data_dir'] if dataset_type == 'clean' else str(os.path.join(os.path.dirname(CONFIG['data_dir']), 'datasets_masked'))
    
    preprocessing_function = None
    if 'densenet' in model_name:
        from tensorflow.keras.applications.densenet import preprocess_input
        preprocessing_function = preprocess_input
    elif 'efficientnet' in model_name:
        from tensorflow.keras.applications.efficientnet import preprocess_input
        preprocessing_function = preprocess_input

    def custom_preprocessing(img):
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        img_clahe = apply_clahe(img)
        img_preprocessed = preprocessing_function(img_clahe)
        return img_preprocessed

    loader = KerasDataLoader()
    _, test_gen, _ = loader.get_train_test_generators(data_dir, CONFIG['img_size'], CONFIG['batch_size'], preprocessing_fn=custom_preprocessing)

    # Load model
    model_path = f'/Users/yaman/master-reseach/models/{model_name}_optimized.keras'
    custom_objects = {
        'focal_loss_fn': FocalLoss(gamma=2.0, alpha=0.25),
        'CBAM': CBAM
    }
    model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    # Predict
    y_true = test_gen.classes
    y_pred_prob = model.predict(test_gen)
    y_pred = np.argmax(y_pred_prob, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Benign', 'Malignant', 'Normal']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    out_path = f'/Users/yaman/master-reseach/results/{model_name}_cm.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    evaluate_and_plot_cm('efficientnetb0')
    evaluate_and_plot_cm('densenet121')
