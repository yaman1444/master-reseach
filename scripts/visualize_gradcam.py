"""
Grad-CAM Visualization for Model Interpretability
Paper: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks" ICCV'17

Formula: L^c_Grad-CAM = ReLU(Σ_k α^c_k A_k)
where α^c_k = (1/Z) Σ_i Σ_j (∂y^c/∂A^k_ij) - global average pooling of gradients
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os

class GradCAM:
    """
    Grad-CAM implementation for CNN visualization
    Highlights discriminative regions for class prediction
    """
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        
        # Create gradient model
        self.grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )
    
    def _find_target_layer(self):
        """Find last convolutional layer automatically"""
        for layer in reversed(self.model.layers):
            # Check if layer has 4D output (batch, height, width, channels)
            if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply Grad-CAM.")
    
    def compute_heatmap(self, image, class_idx=None, eps=1e-8):
        """
        Compute Grad-CAM heatmap
        
        Args:
            image: Input image (preprocessed)
            class_idx: Target class index (None = predicted class)
            eps: Small constant for numerical stability
        
        Returns:
            heatmap: Grad-CAM heatmap (H x W)
        """
        # Compute gradients
        with tf.GradientTape() as tape:
            # Forward pass
            conv_outputs, predictions = self.grad_model(image)
            
            # Get target class
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Class score
            class_channel = predictions[:, class_idx]
        
        # Compute gradients: ∂y^c/∂A^k
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients: α^c_k = (1/Z) Σ_i Σ_j (∂y^c/∂A^k_ij)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps: Σ_k α^c_k A_k
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Average over all feature maps and apply ReLU
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)  # ReLU
        
        # Normalize to [0, 1]
        heatmap = heatmap / (np.max(heatmap) + eps)
        
        return heatmap
    
    def overlay_heatmap(self, heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            heatmap: Grad-CAM heatmap
            image: Original image (0-255)
            alpha: Transparency of heatmap
            colormap: OpenCV colormap
        
        Returns:
            superimposed: Overlayed image
        """
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        
        # Convert to RGB (OpenCV uses BGR)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure image is uint8
        if image.max() <= 1.0:
            image = np.uint8(255 * image)
        
        # Superimpose
        superimposed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return superimposed

def visualize_gradcam_batch(model, images, true_labels, pred_labels, class_names, 
                           save_path='results/gradcam_examples.png', n_samples=9):
    """
    Visualize Grad-CAM for multiple samples
    
    Args:
        model: Trained model
        images: Batch of images
        true_labels: True class indices
        pred_labels: Predicted class indices
        class_names: List of class names
        n_samples: Number of samples to visualize
    """
    gradcam = GradCAM(model)
    
    n_samples = min(n_samples, len(images))
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for idx in range(n_samples):
        # Get image
        img = images[idx]
        
        # Expand dims for model input
        img_input = np.expand_dims(img, axis=0)
        
        # Compute heatmap
        heatmap = gradcam.compute_heatmap(img_input, class_idx=pred_labels[idx])
        
        # Overlay
        superimposed = gradcam.overlay_heatmap(heatmap, img)
        
        # Plot
        ax = axes[idx]
        ax.imshow(superimposed)
        
        true_class = class_names[true_labels[idx]]
        pred_class = class_names[pred_labels[idx]]
        correct = "✓" if true_labels[idx] == pred_labels[idx] else "✗"
        
        ax.set_title(f'{correct} True: {true_class}\nPred: {pred_class}', 
                    fontsize=10, color='green' if correct == "✓" else 'red')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grad-CAM visualizations saved to {save_path}")

def visualize_feature_maps(model, image, layer_names=None, save_path='results/feature_maps.png'):
    """
    Visualize intermediate feature maps evolution
    
    Args:
        model: Trained model
        image: Input image
        layer_names: List of layer names to visualize (None = auto-select)
    """
    if layer_names is None:
        # Auto-select convolutional layers
        layer_names = []
        for layer in model.layers:
            if 'conv' in layer.name.lower() and len(layer.output_shape) == 4:
                layer_names.append(layer.name)
        
        # Select evenly spaced layers (max 6)
        if len(layer_names) > 6:
            indices = np.linspace(0, len(layer_names)-1, 6, dtype=int)
            layer_names = [layer_names[i] for i in indices]
    
    # Create feature extraction model
    outputs = [model.get_layer(name).output for name in layer_names]
    feature_model = Model(inputs=model.input, outputs=outputs)
    
    # Extract features
    img_input = np.expand_dims(image, axis=0)
    feature_maps = feature_model.predict(img_input, verbose=0)
    
    # Plot
    n_layers = len(layer_names)
    fig, axes = plt.subplots(2, n_layers, figsize=(3*n_layers, 6))
    
    for i, (layer_name, fmap) in enumerate(zip(layer_names, feature_maps)):
        # Select first 8 channels
        n_channels = min(8, fmap.shape[-1])
        
        # Average over channels for top row
        avg_fmap = np.mean(fmap[0, :, :, :n_channels], axis=-1)
        axes[0, i].imshow(avg_fmap, cmap='viridis')
        axes[0, i].set_title(f'{layer_name}\n(avg)', fontsize=8)
        axes[0, i].axis('off')
        
        # Show first channel for bottom row
        axes[1, i].imshow(fmap[0, :, :, 0], cmap='viridis')
        axes[1, i].set_title(f'Channel 0', fontsize=8)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature maps saved to {save_path}")

def generate_gradcam_report(model_path, data_dir, class_names, n_samples=12):
    """
    Generate comprehensive Grad-CAM report for a trained model
    
    Args:
        model_path: Path to saved model
        data_dir: Directory with test images
        class_names: List of class names
        n_samples: Number of samples to visualize
    """
    print(f"\nGenerating Grad-CAM visualizations for {model_path}...")
    
    # Load model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Load test data
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=n_samples,
        class_mode='categorical',
        shuffle=True
    )
    
    # Get batch
    images, labels = next(generator)
    true_labels = np.argmax(labels, axis=1)
    
    # Predict
    predictions = model.predict(images, verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    
    # Generate visualizations
    model_name = os.path.basename(model_path).replace('.keras', '')
    
    visualize_gradcam_batch(
        model, images, true_labels, pred_labels, class_names,
        save_path=f'results/{model_name}_gradcam.png',
        n_samples=n_samples
    )
    
    # Feature maps for first image
    visualize_feature_maps(
        model, images[0],
        save_path=f'results/{model_name}_feature_maps.png'
    )
    
    print(f"Visualizations complete for {model_name}")

if __name__ == '__main__':
    # Example usage
    os.makedirs('results', exist_ok=True)
    
    class_names = ['benign', 'malignant', 'normal']
    
    # Generate for all trained models
    model_files = [
        'models/densenet121_final.keras',
        'models/resnet50_final.keras',
        'models/efficientnetb0_final.keras'
    ]
    
    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                generate_gradcam_report(
                    model_path=model_path,
                    data_dir='../datasets/test/',
                    class_names=class_names,
                    n_samples=12
                )
            except Exception as e:
                print(f"Error processing {model_path}: {e}")
        else:
            print(f"Model not found: {model_path}")
