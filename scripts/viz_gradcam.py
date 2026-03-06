import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Custom objects and imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from focal_loss import FocalLoss
from cbam import CBAM

TEST_DIR = '../datasets_split/test/'
MODEL_PATH = 'models/densenet121_final.keras'
IMG_SIZE = (320, 320)
CLASS_NAMES = ['debut', 'grave', 'normal']

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_gradcam_viz(img_path, heatmap, save_path, class_name, pred_name, prob):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    # We rescale heatmap to a range 0-255
    heatmap_display = np.uint8(255 * heatmap)
    # Use jet colormap to colorize heatmap
    jet = plt.colormaps.get_cmap("jet")
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_display]

    # Create an image with RGB colorized heatmap
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    superimposed_img = jet_heatmap * 0.4 + (img / 255.0) * 0.6
    
    superimposed_img = np.clip(superimposed_img, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title(f"Image Originale\nClasse Réelle : {class_name}", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(superimposed_img)
    title_color = 'green' if class_name == pred_name else 'red'
    axes[1].set_title(f"Grad-CAM (Attention du Modèle)\nPrédiction : {pred_name} ({(prob*100):.1f}%)", 
                      fontsize=14, fontweight='bold', color=title_color)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def load_model_safe(model_path):
    custom_objects = {
        'FocalLoss': FocalLoss,
        'CBAM': CBAM,
        'ChannelAttention': __import__('cbam', fromlist=['ChannelAttention']).ChannelAttention,
        'SpatialAttention': __import__('cbam', fromlist=['SpatialAttention']).SpatialAttention,
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


print("📦 Loading model...")
model = load_model_safe(MODEL_PATH)
last_conv_layer_name = "conv5_block16_concat" # For DenseNet121

os.makedirs('results/presentation/gradcam', exist_ok=True)

for i, class_name in enumerate(CLASS_NAMES):
    class_dir = os.path.join(TEST_DIR, class_name)
    if not os.path.exists(class_dir): continue
    
    # Pick the first image
    img_name = os.listdir(class_dir)[0]
    img_path = os.path.join(class_dir, img_name)
    
    # Prep image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img_array = img.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(preds[0])
    pred_name = CLASS_NAMES[pred_idx]
    prob = preds[0][pred_idx]
    
    # GradCAM
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    # Save
    save_path = f"results/presentation/gradcam/7_gradcam_{class_name}.png"
    save_gradcam_viz(img_path, heatmap, save_path, class_name, pred_name, prob)
    print(f"✅ Grad-CAM géré pour {class_name} -> {save_path}")

print("✅ Toutes les visualisations de présentation sont prêtes dans 'results/presentation/' !")
