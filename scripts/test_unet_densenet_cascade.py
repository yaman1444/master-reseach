import os
import cv2
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from cbam import CBAM
from focal_loss import FocalLoss
from augmentation import apply_clahe
from train_unet import dice_loss, dice_coef
import matplotlib.pyplot as plt

def apply_clahe_np(img):
    clahe_img = apply_clahe(img)
    return clahe_img

def test_cascade(test_dir, unet_model, densenet_model, class_names):
    y_true = []
    y_pred = []
    
    # Process test data
    save_examples_dir = 'results/cascade_examples'
    os.makedirs(save_examples_dir, exist_ok=True)
    
    examples_saved = 0
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.exists(class_path):
            continue
            
        for file in os.listdir(class_path):
            if file.endswith('.png') and '_mask' not in file:
                img_path = os.path.join(class_path, file)
                
                # 1. Load image
                original_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if original_img is None: continue
                
                # 2. UNet Inference (256x256)
                unet_input = cv2.resize(original_img, (256, 256))
                unet_input = unet_input / 255.0
                unet_input_batch = np.expand_dims(unet_input, axis=0)
                
                mask_pred = unet_model.predict(unet_input_batch, verbose=0)[0]
                mask_binary = (mask_pred > 0.5).astype(np.float32)
                
                # If the mask is entirely empty and it's not the "normal" class, 
                # or even if it is, we might lose all info. 
                # For safety, if mask is empty, we can just use the original image, or stick to the strict cascade.
                # Let's stick to strict cascade: black out background.
                
                # 3. DenseNet Inference (224x224)
                mask_densenet = cv2.resize(mask_binary, (224, 224))
                mask_densenet = np.expand_dims(mask_densenet, axis=-1)
                
                img_densenet = cv2.resize(original_img, (224, 224))
                img_densenet = img_densenet / 255.0
                
                img_clahe = apply_clahe_np(img_densenet)
                masked_img = img_clahe * mask_densenet
                
                densenet_input = np.expand_dims(masked_img, axis=0)
                pred_probs = densenet_model.predict(densenet_input, verbose=0)[0]
                
                # Use simple argmax for this test, or we could load thresholds
                # For simplicity in this script, we'll use argmax as a baseline
                pred_class_idx = np.argmax(pred_probs)
                
                y_true.append(class_idx)
                y_pred.append(pred_class_idx)
                
                # Save a few examples
                if examples_saved < 5 and class_name != 'normal':
                    plt.figure(figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.title("Original (CLAHE)")
                    plt.imshow(img_clahe)
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 2)
                    plt.imshow(mask_densenet[:, :, 0], cmap='gray')
                    
                    plt.subplot(1, 3, 3)
                    plt.title(f"Masked Input (True: {class_name}, Pred: {class_names[pred_class_idx]})")
                    plt.imshow(masked_img)
                    plt.axis('off')
                    
                    plt.savefig(os.path.join(save_examples_dir, f'cascade_ex_{examples_saved}.png'))
                    plt.close()
                    examples_saved += 1
                    
    print("\n=== Résultats de la Cascade U-Net + DenseNet ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    print("Matrice de Confusion :")
    print(cm)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"\nAccuracy globale: {acc:.4f}")
    print(f"F1-Score (macro): {f1:.4f}")


if __name__ == '__main__':
    unet_path = 'models/unet_segmentation.keras'
    densenet_path = 'models/densenet121_final.keras'
    
    if not os.path.exists(densenet_path):
        # Fallback to whatever densenet we have
        densenet_path = 'scripts/my_model.keras'
        
    print("Loading U-Net...")
    unet = tf.keras.models.load_model(unet_path, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})
    
    print(f"Loading DenseNet from {densenet_path}...")
    custom_objects = {'CBAM': CBAM, 'FocalLoss': FocalLoss}
    densenet = tf.keras.models.load_model(densenet_path, custom_objects=custom_objects)
    
    class_names = ['normal', 'debut', 'grave']
    test_dir = 'datasets_split/test'
    
    test_cascade(test_dir, unet, densenet, class_names)
