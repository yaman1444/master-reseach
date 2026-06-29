"""
Demo: Single Image Prediction with Grad-CAM Visualization
Usage: python demo_predict.py --image path/to/image.png --model models/densenet121_final.keras
"""
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
import os

from visualize_gradcam import GradCAM
from cbam import CBAM  # Import custom layer
from focal_loss import FocalLoss  # Import custom loss

def predict_with_gradcam(model_path, image_path, class_names=['benign', 'malignant', 'normal'], 
                        use_calibrated_threshold=False, threshold_config=None):
    """
    Predict class and generate Grad-CAM visualization
    
    Args:
        model_path: Path to trained model
        image_path: Path to input image
        class_names: List of class names
        use_calibrated_threshold: Use calibrated threshold for malignant class
        threshold_config: Dict with 'optimal_threshold_malignant' key
    
    Returns:
        prediction: Predicted class name
        confidence: Prediction confidence
        gradcam_overlay: Grad-CAM overlay image
    """
    # Load model
    print(f"Loading model from {model_path}...")
    # Register custom objects
    custom_objects = {'CBAM': CBAM, 'FocalLoss': FocalLoss}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    
    # Load and preprocess image
    print(f"Loading image from {image_path}...")
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array_input = np.expand_dims(img_array, axis=0)
    
    # Predict
    print("Predicting...")
    predictions = model.predict(img_array_input, verbose=0)
    
    # Apply calibrated threshold if requested
    if use_calibrated_threshold and threshold_config:
        malignant_idx = 1  # Assuming malignant is index 1
        threshold = threshold_config.get('optimal_threshold_malignant', 0.5)
        
        if predictions[0][malignant_idx] >= threshold:
            predicted_class_idx = malignant_idx
            print(f"⚠️  Calibrated threshold applied: malignant prob={predictions[0][malignant_idx]:.3f} >= {threshold:.3f}")
        else:
            predicted_class_idx = np.argmax(predictions[0])
    else:
        predicted_class_idx = np.argmax(predictions[0])
    
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Generate Grad-CAM
    print("Generating Grad-CAM...")
    try:
        # Try to find a convolutional layer
        gradcam = GradCAM(model, layer_name='relu')  # DenseNet121's last activation
        heatmap = gradcam.compute_heatmap(img_array_input, class_idx=predicted_class_idx)
        
        # Overlay heatmap
        img_uint8 = (img_array * 255).astype(np.uint8)
        gradcam_overlay = gradcam.overlay_heatmap(heatmap, img_uint8, alpha=0.4)
    except Exception as e:
        print(f"Warning: Grad-CAM failed ({e}), using original image")
        gradcam_overlay = (img_array * 255).astype(np.uint8)
    
    return predicted_class, confidence, predictions[0], gradcam_overlay, img_array

def visualize_prediction(image_path, predicted_class, confidence, probabilities, 
                        gradcam_overlay, original_img, class_names, save_path=None):
    """
    Create comprehensive visualization of prediction
    """
    fig = plt.figure(figsize=(16, 6))
    
    # Original image
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(original_img)
    ax1.set_title('Original Image', fontsize=14)
    ax1.axis('off')
    
    # Grad-CAM overlay
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(gradcam_overlay)
    ax2.set_title(f'Grad-CAM: {predicted_class.upper()}\nConfidence: {confidence:.2f}%', 
                 fontsize=14, color='green' if confidence > 90 else 'orange')
    ax2.axis('off')
    
    # Probability bar chart
    ax3 = plt.subplot(1, 3, 3)
    colors = ['green' if i == np.argmax(probabilities) else 'gray' 
             for i in range(len(class_names))]
    bars = ax3.barh(class_names, probabilities * 100, color=colors, alpha=0.7)
    ax3.set_xlabel('Probability (%)', fontsize=12)
    ax3.set_title('Class Probabilities', fontsize=14)
    ax3.set_xlim([0, 100])
    
    # Add value labels
    for bar, prob in zip(bars, probabilities):
        width = bar.get_width()
        ax3.text(width + 2, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.2f}%', ha='left', va='center', fontsize=10)
    
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Breast Cancer Classification Demo')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='models/densenet121_final.keras',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save visualization (optional)')
    parser.add_argument('--classes', type=str, nargs='+', 
                       default=['benign', 'malignant', 'normal'],
                       help='Class names')
    parser.add_argument('--use_calibrated', action='store_true',
                       help='Use calibrated threshold for malignant class')
    parser.add_argument('--threshold_config', type=str, default='results/densenet121_improved_thresholds.json',
                       help='Path to threshold configuration JSON')
    
    args = parser.parse_args()
    
    # Load threshold config if using calibrated
    threshold_config = None
    if args.use_calibrated:
        try:
            import json
            with open(args.threshold_config, 'r') as f:
                threshold_config = json.load(f)
            print(f"✅ Loaded calibrated thresholds from {args.threshold_config}")
        except:
            print(f"⚠️  Could not load {args.threshold_config}, using standard argmax")
            args.use_calibrated = False
    
    # Check files exist
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        return
    
    print("\n" + "="*80)
    print("BREAST CANCER CLASSIFICATION DEMO")
    print("="*80 + "\n")
    
    # Predict
    predicted_class, confidence, probabilities, gradcam_overlay, original_img = \
        predict_with_gradcam(args.model, args.image, args.classes, 
                           args.use_calibrated, threshold_config)
    
    # Print results
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    print(f"\nPredicted Class: {predicted_class.upper()}")
    print(f"Confidence: {confidence:.2f}%\n")
    print("Class Probabilities:")
    for class_name, prob in zip(args.classes, probabilities):
        print(f"  {class_name.capitalize()}: {prob*100:.2f}%")
    print("\n" + "="*80 + "\n")
    
    # Visualize
    output_path = args.output or f'results/prediction_{os.path.basename(args.image)}'
    visualize_prediction(
        args.image, predicted_class, confidence, probabilities,
        gradcam_overlay, original_img, args.classes, save_path=output_path
    )
    
    # Clinical interpretation
    print("\n" + "="*80)
    print("CLINICAL INTERPRETATION")
    print("="*80)
    
    if predicted_class == 'malignant':
        if confidence > 90:
            print("⚠️  HIGH CONFIDENCE malignant prediction")
            print("    Recommendation: Immediate biopsy and specialist consultation")
        else:
            print("⚠️  MODERATE CONFIDENCE malignant prediction")
            print("    Recommendation: Additional imaging and specialist review")
    elif predicted_class == 'benign':
        if confidence > 90:
            print("✓  HIGH CONFIDENCE benign prediction")
            print("    Recommendation: Routine follow-up monitoring")
        else:
            print("⚠️  MODERATE CONFIDENCE benign prediction")
            print("    Recommendation: Short-term follow-up imaging")
    else:  # normal
        if confidence > 90:
            print("✓  HIGH CONFIDENCE normal tissue")
            print("    Recommendation: Routine screening schedule")
        else:
            print("⚠️  MODERATE CONFIDENCE normal prediction")
            print("    Recommendation: Consider additional views")
    
    print("\n⚠️  DISCLAIMER: This is an AI-assisted tool for research purposes only.")
    print("    All predictions must be verified by qualified medical professionals.")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
