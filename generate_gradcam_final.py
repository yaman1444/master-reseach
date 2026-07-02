import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2

sys.path.append('/Users/yaman/master-reseach/src')
from domain.preprocessing import apply_clahe
from domain.focal_loss import FocalLoss
from domain.cbam import CBAM

class GradCAM:
    def __init__(self, model, base_model_name, last_conv_layer_name):
        self.model = model
        base_model = model.get_layer(base_model_name)
        self.last_conv_layer = base_model.get_layer(last_conv_layer_name)
        
        self.grad_model = tf.keras.models.Model(
            inputs=[model.input], 
            outputs=[self.last_conv_layer.output, model.output]
        )
        
    def compute_heatmap(self, image, class_idx, eps=1e-8):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            loss = predictions[:, class_idx]
            
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        image = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(cv2.resize(heatmap, (image.shape[1], image.shape[0])), colormap)
        superimposed_img = heatmap * alpha + image * (1 - alpha)
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        return superimposed_img

def generate_images():
    print("Loading model...")
    custom_objects = {
        'focal_loss_fn': FocalLoss(gamma=2.0, alpha=0.25),
        'CBAM': CBAM
    }
    model = keras.models.load_model('/Users/yaman/master-reseach/models/efficientnetb0_optimized.keras', 
                                  custom_objects=custom_objects, compile=False)
    
    gradcam = GradCAM(model, 'efficientnetb0', 'top_activation')
    
    from tensorflow.keras.applications.efficientnet import preprocess_input
    
    data_dir = '/Users/yaman/master-reseach/datasets/train'
    classes = [('benign', 'Bénin', 0), ('malignant', 'Malin', 1), ('normal', 'Normal', 2)]
    
    for class_folder, class_name_fr, class_idx in classes:
        folder_path = os.path.join(data_dir, class_folder)
        img_names = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        
        # Select a deterministic image for consistency
        img_path = os.path.join(folder_path, img_names[5]) 
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (224, 224))
        
        img_clahe = apply_clahe(img_resized)
        img_preprocessed = preprocess_input(img_clahe.astype(np.float32))
        img_input = np.expand_dims(img_preprocessed, axis=0)
        
        pred_probs = model.predict(img_input)
        pred_idx = np.argmax(pred_probs[0])
        pred_class_fr = classes[pred_idx][1]
        
        heatmap = gradcam.compute_heatmap(img_input, class_idx)
        superimposed = gradcam.overlay_heatmap(heatmap, img_resized)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(superimposed)
        
        correct = "✓" if class_idx == pred_idx else "✗"
        color = 'green' if class_idx == pred_idx else 'red'
        ax.set_title(f'{correct} True: {class_name_fr} | Pred: {pred_class_fr}', fontsize=16, color=color, fontweight='bold')
        ax.axis('off')
        
        out_path = f'/Users/yaman/master-reseach/results/gradcam_{class_folder}.png'
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved {out_path}")

if __name__ == '__main__':
    generate_images()
