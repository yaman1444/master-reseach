"""
Grad-CAM++ for Advanced Classification Head
=============================================
Visualizes model attention (heatmaps) over images.
Supports the DenseNet121+CBAM+BatchNorm custom head.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import cv2


def get_last_conv_layer_name(model):
    """Finds the last convolutional layer in the backbone."""
    # Since our custom model wraps the backend, we need to find the base model
    # The first layer represents the input, and the second is the backbone
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        # Maybe it's not wrapped? Let's search inside the model itself
        base_model = model

    for layer in reversed(base_model.layers):
        # DenseNet specific
        if layer.name == 'relu':
            return layer.name, base_model
        # Check if layer has output_shape attribute
        if hasattr(layer, 'output_shape'):
            if isinstance(layer.output_shape, tuple) and len(layer.output_shape) == 4 and 'conv' in layer.name.lower():
                return layer.name, base_model
        # Alternative shape check for newer TF versions
        elif hasattr(layer, 'output') and hasattr(layer.output, 'shape'):
            if len(layer.output.shape) == 4 and 'conv' in layer.name.lower():
                return layer.name, base_model

    raise ValueError("Impossible de trouver la dernière couche de convolution.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates Grad-CAM++ heatmap."""
    # Find base model and specific layers
    base_model = None
    for l in model.layers:
        if isinstance(l, tf.keras.Model):
            base_model = l
            break
            
    if not base_model:
        base_model = model

    # Create a model with multiple outputs
    # 1. Output from the target conv layer
    # 2. Final prediction output
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    
    # We need to build a new model that connects input -> conv -> output
    # Since we have a custom head with CBAM, we just use tf.GradientTape directly
    # on the full model architecture.
    
    # 1st Model: map input to last conv layer
    grad_model_1 = tf.keras.models.Model(
        [base_model.inputs], [last_conv_layer.output]
    )
    
    # 2nd Model: map last conv layer to predictions
    # This is tricky with custom heads. We'll trace the graph manually.
    
    with tf.GradientTape() as tape:
        # Get conv outputs
        conv_outputs = grad_model_1(img_array)
        tape.watch(conv_outputs)
        
        # Manually pass through the rest of the base model (if any layers remain)
        x = conv_outputs
        in_base = False
        for layer in base_model.layers:
            if layer.name == last_conv_layer_name:
                in_base = True
                continue
            if in_base:
                x = layer(x)
                
        # Now pass through the custom head
        head_started = False
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                head_started = True
                continue
            if head_started:
                x = layer(x)
                
        preds = x
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradients
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Grad-CAM++ logic
    # Calculate first, second, and third derivatives
    first_derivative = tf.exp(class_channel)[0] * grads
    second_derivative = first_derivative * grads
    third_derivative = second_derivative * grads
    
    global_sum = tf.reduce_sum(conv_outputs[0], axis=(0, 1))
    alpha_num = second_derivative[0]
    alpha_denom = second_derivative[0] * 2.0 + third_derivative[0] * global_sum
    # Avoid zero division
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))
    
    alphas = alpha_num / alpha_denom
    weights = tf.maximum(first_derivative[0], 0.0) # ReLU
    deep_linearization_weights = tf.reduce_sum(weights * alphas, axis=(0, 1))

    # Apply weights to conv outputs
    cam = tf.reduce_sum(tf.multiply(deep_linearization_weights, conv_outputs[0]), axis=-1)
    
    # ReLU on CAM (we only care about positive influences)
    heatmap = tf.maximum(cam, 0) / tf.math.reduce_max(cam)
    return heatmap.numpy(), pred_index


def save_and_display_gradcam(img_path, heatmap, model_name,
                             pred_class, true_class, alpha=0.4):
    """Overlays heatmap on original image and saves it."""
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (320, 320))
    
    # Resize heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap_colored = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    # Superimpose
    superimposed_img = heatmap_colored * alpha + img * (1-alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    # Create side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Original (True: {true_class})')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Grad-CAM++ (Pred: {pred_class})')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    os.makedirs('results/gradcam', exist_ok=True)
    out_path = f'results/gradcam/gradcam_{os.path.basename(img_path)}'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return out_path


def main():
    print("🚀 Démarrage de Grad-CAM++...")
    
    # 1. Load Custom Model
    model_path = 'models/densenet121_final.keras'
    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable: {model_path}")
        return
        
    print(f"📦 Chargement du modèle {model_path}")
    
    from cbam import CBAM
    from focal_loss import FocalLoss
    custom_objects = {'CBAM': CBAM, 'FocalLoss': FocalLoss}
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    class_names = ['debut', 'grave', 'normal']
    
    # Find layer
    try:
        layer_name, base_model = get_last_conv_layer_name(model)
        print(f"🎯 Couche cible trouvée: {layer_name}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return

    # 2. Run on positive test examples ('grave' = malignant)
    test_dir = '../datasets_split/test/grave/'
    
    if not os.path.exists(test_dir):
         print(f"❌ Dossier introuvable: {test_dir}")
         return
         
    files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))][:5]
    print(f"🔍 Analyse de {len(files)} images malignes...")
    
    for filename in files:
        img_path = os.path.join(test_dir, filename)
        
        # Preprocess
        img = load_img(img_path, target_size=(320, 320))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Same as training
        
        # Get True prediction
        preds = model.predict(img_array, verbose=0)
        pred_idx = np.argmax(preds[0])
        pred_class = class_names[pred_idx]
        conf = preds[0][pred_idx]
        
        print(f"  ➜ {filename} : Prédit '{pred_class}' ({conf:.1%})")
        
        # Only process if correctly predicted as 'grave'
        if pred_class == 'grave':
            try:
                heatmap, _ = make_gradcam_heatmap(img_array, model, layer_name, pred_index=1)
                
                out_path = save_and_display_gradcam(
                    img_path, heatmap, 'densenet121', 
                    f"{pred_class} ({conf:.1%})", 'grave'
                )
                print(f"      ✅ Heatmap générée: {out_path}")
            except Exception as e:
                print(f"      ❌ Échec de la Heatmap: {e}")


if __name__ == '__main__':
    main()
