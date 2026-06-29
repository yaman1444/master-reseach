"""
Comprehensive Visualization Suite
- t-SNE/UMAP embeddings for feature space visualization
- Confusion matrix heatmaps
- SHAP analysis for global feature importance
- ROC curves per class
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import os

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

def extract_features(model, data_generator, layer_name=None):
    """
    Extract features from intermediate layer for embedding visualization
    
    Args:
        model: Trained model
        data_generator: Data generator
        layer_name: Layer to extract features from (None = penultimate layer)
    
    Returns:
        features: Extracted features (N x D)
        labels: True labels (N,)
    """
    if layer_name is None:
        # Use layer before final classification
        for layer in reversed(model.layers):
            if 'dense' in layer.name.lower() and layer != model.layers[-1]:
                layer_name = layer.name
                break
        
        if layer_name is None:
            layer_name = model.layers[-2].name
    
    print(f"Extracting features from layer: {layer_name}")
    
    # Create feature extraction model
    feature_model = Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    
    # Extract features
    data_generator.reset()
    features = feature_model.predict(data_generator, verbose=1)
    labels = data_generator.classes
    
    return features, labels

def plot_tsne_umap(features, labels, class_names, save_path='results/embeddings.png'):
    """
    Plot t-SNE and UMAP embeddings side by side
    
    t-SNE: van der Maaten & Hinton, JMLR 2008
    UMAP: McInnes et al., arXiv 2018
    """
    fig, axes = plt.subplots(1, 2 if UMAP_AVAILABLE else 1, figsize=(16, 6))
    
    if not UMAP_AVAILABLE:
        axes = [axes]
    
    # t-SNE
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_tsne = tsne.fit_transform(features)
    
    ax = axes[0]
    scatter = ax.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                        c=labels, cmap='viridis', alpha=0.6, s=20)
    ax.set_title('t-SNE Embedding', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                         markerfacecolor=scatter.cmap(scatter.norm(i)), 
                         markersize=10, label=class_names[i])
              for i in range(len(class_names))]
    ax.legend(handles=handles, loc='best')
    ax.grid(True, alpha=0.3)
    
    # UMAP
    if UMAP_AVAILABLE:
        print("Computing UMAP embedding...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        features_umap = reducer.fit_transform(features)
        
        ax = axes[1]
        scatter = ax.scatter(features_umap[:, 0], features_umap[:, 1], 
                           c=labels, cmap='viridis', alpha=0.6, s=20)
        ax.set_title('UMAP Embedding', fontsize=14)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=scatter.cmap(scatter.norm(i)), 
                             markersize=10, label=class_names[i])
                  for i in range(len(class_names))]
        ax.legend(handles=handles, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Embeddings saved to {save_path}")

def plot_roc_curves(model, data_generator, class_names, save_path='results/roc_curves.png'):
    """
    Plot ROC curves for each class (one-vs-rest)
    """
    # Get predictions
    data_generator.reset()
    y_pred_proba = model.predict(data_generator, verbose=0)
    y_true = tf.keras.utils.to_categorical(data_generator.classes, len(class_names))
    
    # Compute ROC curve for each class
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
               label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curves saved to {save_path}")

def plot_confusion_matrix_detailed(model, data_generator, class_names, 
                                   save_path='results/confusion_matrix_detailed.png'):
    """
    Plot detailed confusion matrix with percentages
    """
    # Get predictions
    data_generator.reset()
    y_pred_proba = model.predict(data_generator, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = data_generator.classes
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Absolute counts
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names,
               ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title('Confusion Matrix (Counts)', fontsize=14)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    # Normalized percentages
    ax = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               ax=ax, cbar_kws={'label': 'Percentage'})
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed confusion matrix saved to {save_path}")

def shap_analysis(model, data_generator, class_names, n_samples=100,
                 save_path='results/shap_summary.png'):
    """
    SHAP (SHapley Additive exPlanations) analysis for global feature importance
    
    Paper: Lundberg & Lee, NeurIPS 2017
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available. Skipping SHAP analysis.")
        return
    
    print("Running SHAP analysis (this may take a while)...")
    
    # Get sample data
    data_generator.reset()
    images, labels = [], []
    
    for i in range(min(n_samples // data_generator.batch_size + 1, len(data_generator))):
        batch_x, batch_y = next(data_generator)
        images.append(batch_x)
        labels.append(batch_y)
    
    images = np.vstack(images)[:n_samples]
    labels = np.vstack(labels)[:n_samples]
    
    # Create explainer
    background = images[:10]  # Use subset as background
    explainer = shap.DeepExplainer(model, background)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(images[:50])  # Analyze 50 samples
    
    # Plot summary
    fig, axes = plt.subplots(1, len(class_names), figsize=(6*len(class_names), 6))
    
    if len(class_names) == 1:
        axes = [axes]
    
    for i, class_name in enumerate(class_names):
        ax = axes[i]
        
        # Average absolute SHAP values across samples
        mean_shap = np.mean(np.abs(shap_values[i]), axis=0)
        
        # Plot heatmap
        im = ax.imshow(mean_shap, cmap='hot', aspect='auto')
        ax.set_title(f'SHAP Importance: {class_name}', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SHAP analysis saved to {save_path}")

def generate_all_visualizations(model_path, data_dir, class_names):
    """
    Generate all visualizations for a trained model
    
    Args:
        model_path: Path to saved model
        data_dir: Directory with test images
        class_names: List of class names
    """
    print(f"\n{'='*80}")
    print(f"Generating visualizations for {os.path.basename(model_path)}")
    print(f"{'='*80}\n")
    
    # Load model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Load data
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    model_name = os.path.basename(model_path).replace('.keras', '')
    
    # 1. Extract features and plot embeddings
    print("\n1. Extracting features for embeddings...")
    features, labels = extract_features(model, generator)
    plot_tsne_umap(features, labels, class_names, 
                   save_path=f'results/{model_name}_embeddings.png')
    
    # 2. ROC curves
    print("\n2. Plotting ROC curves...")
    plot_roc_curves(model, generator, class_names,
                   save_path=f'results/{model_name}_roc_curves.png')
    
    # 3. Detailed confusion matrix
    print("\n3. Plotting confusion matrix...")
    plot_confusion_matrix_detailed(model, generator, class_names,
                                   save_path=f'results/{model_name}_confusion_detailed.png')
    
    # 4. SHAP analysis
    print("\n4. Running SHAP analysis...")
    shap_analysis(model, generator, class_names,
                 save_path=f'results/{model_name}_shap.png')
    
    print(f"\n{'='*80}")
    print(f"All visualizations complete for {model_name}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    
    class_names = ['benign', 'malignant', 'normal']
    
    # Generate visualizations for all trained models
    model_files = [
        'models/densenet121_final.keras',
        'models/resnet50_final.keras',
        'models/efficientnetb0_final.keras'
    ]
    
    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                generate_all_visualizations(
                    model_path=model_path,
                    data_dir='../datasets/test/',
                    class_names=class_names
                )
            except Exception as e:
                print(f"Error processing {model_path}: {e}")
        else:
            print(f"Model not found: {model_path}")
