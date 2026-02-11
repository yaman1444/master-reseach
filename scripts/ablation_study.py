"""
Ablation Study: Systematic evaluation of each component's contribution
Tests: baseline, +augmentation, +dropout, +CBAM, +ensemble

Goal: Identify which components provide the most value (e.g., +5% F1 where ResNet overfits)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from train_advanced import train_model, CONFIG

def run_ablation_study(train_dir='../datasets/train/', val_dir='../datasets/test/'):
    """
    Run systematic ablation study
    
    Configurations tested:
    1. Baseline: No augmentation, no dropout, no CBAM
    2. +Augmentation: Add CLAHE + Mixup
    3. +Dropout: Add dropout layers
    4. +CBAM: Add attention mechanism
    5. Full: All components
    """
    
    ablation_configs = {
        'baseline': {
            'use_clahe': False,
            'use_mixup': False,
            'dropout_rate': 0.0,
            'use_cbam': False,
            'description': 'Baseline (no augmentation, no dropout, no CBAM)'
        },
        'aug_only': {
            'use_clahe': True,
            'use_mixup': True,
            'dropout_rate': 0.0,
            'use_cbam': False,
            'description': 'Baseline + CLAHE + Mixup'
        },
        'aug_dropout': {
            'use_clahe': True,
            'use_mixup': True,
            'dropout_rate': 0.5,
            'use_cbam': False,
            'description': 'Baseline + Augmentation + Dropout(0.5)'
        },
        'aug_dropout_cbam': {
            'use_clahe': True,
            'use_mixup': True,
            'dropout_rate': 0.5,
            'use_cbam': True,
            'description': 'Full model (all components)'
        }
    }
    
    results = {}
    
    for config_name, ablation_config in ablation_configs.items():
        print(f"\n{'='*80}")
        print(f"ABLATION STUDY: {config_name.upper()}")
        print(f"Description: {ablation_config['description']}")
        print(f"{'='*80}\n")
        
        # Update config
        test_config = CONFIG.copy()
        test_config.update({
            'use_clahe': ablation_config['use_clahe'],
            'use_mixup': ablation_config['use_mixup'],
            'dropout_rate': ablation_config['dropout_rate'],
            'use_cbam': ablation_config['use_cbam'],
            'initial_epochs': 10,  # Reduced for ablation
            'fine_tune_epochs': 15
        })
        
        try:
            model, result, history = train_model(
                model_name=f'densenet121_ablation_{config_name}',
                train_dir=train_dir,
                val_dir=val_dir,
                config=test_config
            )
            
            results[config_name] = {
                'config': ablation_config,
                'accuracy': result['accuracy'],
                'macro_f1': result['macro_f1'],
                'classification_report': result['classification_report'],
                'description': ablation_config['description']
            }
            
        except Exception as e:
            print(f"Error in ablation {config_name}: {e}")
            continue
    
    # Generate ablation report
    generate_ablation_report(results)
    
    return results

def generate_ablation_report(results):
    """Generate comprehensive ablation study report"""
    
    # Create comparison table
    rows = []
    for config_name, result in results.items():
        row = {
            'Configuration': result['description'],
            'Accuracy': f"{result['accuracy']:.4f}",
            'Macro-F1': f"{result['macro_f1']:.4f}",
        }
        
        # Add per-class F1
        report = result['classification_report']
        for class_name in ['benign', 'malignant', 'normal']:
            if class_name in report:
                row[f'{class_name.capitalize()}-F1'] = f"{report[class_name]['f1-score']:.4f}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save results
    df.to_csv('results/ablation_study.csv', index=False)
    
    print("\n" + "="*100)
    print("ABLATION STUDY RESULTS")
    print("="*100 + "\n")
    print(df.to_markdown(index=False))
    
    # Save markdown report
    with open('results/ablation_study.md', 'w') as f:
        f.write("# Ablation Study Results\n\n")
        f.write("## Systematic Component Evaluation\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Key Findings\n\n")
        
        # Calculate improvements
        if 'baseline' in results and 'aug_dropout_cbam' in results:
            baseline_f1 = results['baseline']['macro_f1']
            full_f1 = results['aug_dropout_cbam']['macro_f1']
            improvement = (full_f1 - baseline_f1) * 100
            
            f.write(f"- **Total Improvement**: {improvement:.2f}% macro-F1 gain from baseline to full model\n")
            
            # Component-wise gains
            if 'aug_only' in results:
                aug_gain = (results['aug_only']['macro_f1'] - baseline_f1) * 100
                f.write(f"- **Augmentation Contribution**: +{aug_gain:.2f}% macro-F1\n")
            
            if 'aug_dropout' in results and 'aug_only' in results:
                dropout_gain = (results['aug_dropout']['macro_f1'] - results['aug_only']['macro_f1']) * 100
                f.write(f"- **Dropout Contribution**: +{dropout_gain:.2f}% macro-F1\n")
            
            if 'aug_dropout_cbam' in results and 'aug_dropout' in results:
                cbam_gain = (results['aug_dropout_cbam']['macro_f1'] - results['aug_dropout']['macro_f1']) * 100
                f.write(f"- **CBAM Contribution**: +{cbam_gain:.2f}% macro-F1\n")
    
    # Plot ablation results
    plot_ablation_results(results)
    
    print("\nAblation study report saved to results/ablation_study.md")

def plot_ablation_results(results):
    """Plot ablation study results"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    configs = list(results.keys())
    descriptions = [results[c]['description'].split('(')[0].strip() for c in configs]
    accuracies = [results[c]['accuracy'] for c in configs]
    macro_f1s = [results[c]['macro_f1'] for c in configs]
    
    # 1. Overall metrics
    ax = axes[0]
    x = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    bars2 = ax.bar(x + width/2, macro_f1s, width, label='Macro-F1', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Ablation Study: Overall Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(descriptions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Incremental gains
    ax = axes[1]
    
    if len(configs) > 1:
        baseline_f1 = macro_f1s[0]
        gains = [(f1 - baseline_f1) * 100 for f1 in macro_f1s]
        
        colors = ['red' if g < 0 else 'green' for g in gains]
        bars = ax.bar(descriptions, gains, alpha=0.8, color=colors)
        
        ax.set_ylabel('Macro-F1 Gain (%)')
        ax.set_title('Incremental Gains vs Baseline')
        ax.set_xticklabels(descriptions, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, gain in zip(bars, gains):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{gain:+.2f}%', ha='center', 
                   va='bottom' if gain >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/ablation_study_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_ensemble(model_paths, val_dir='../datasets/test/', class_names=None):
    """
    Test ensemble voting (soft voting) across multiple models
    
    Args:
        model_paths: List of paths to trained models
        val_dir: Validation directory
        class_names: List of class names
    """
    print(f"\n{'='*80}")
    print("ENSEMBLE EVALUATION: Soft Voting")
    print(f"Models: {len(model_paths)}")
    print(f"{'='*80}\n")
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import classification_report, accuracy_score
    
    # Load validation data
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    if class_names is None:
        class_names = list(val_generator.class_indices.keys())
    
    # Load models and get predictions
    all_predictions = []
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"Warning: Model not found: {model_path}")
            continue
        
        print(f"Loading {os.path.basename(model_path)}...")
        model = tf.keras.models.load_model(model_path, compile=False)
        
        val_generator.reset()
        predictions = model.predict(val_generator, verbose=0)
        all_predictions.append(predictions)
    
    if len(all_predictions) == 0:
        print("Error: No models loaded successfully")
        return None
    
    # Soft voting: average probabilities
    ensemble_predictions = np.mean(all_predictions, axis=0)
    ensemble_labels = np.argmax(ensemble_predictions, axis=1)
    
    # True labels
    true_labels = val_generator.classes
    
    # Evaluate
    accuracy = accuracy_score(true_labels, ensemble_labels)
    report = classification_report(true_labels, ensemble_labels, 
                                   target_names=class_names, 
                                   output_dict=True, zero_division=0)
    
    print("\nEnsemble Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {report['macro avg']['f1-score']:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, ensemble_labels, 
                               target_names=class_names, zero_division=0))
    
    # Save results
    ensemble_results = {
        'models': [os.path.basename(p) for p in model_paths],
        'accuracy': accuracy,
        'macro_f1': report['macro avg']['f1-score'],
        'classification_report': report
    }
    
    with open('results/ensemble_results.json', 'w') as f:
        json.dump(ensemble_results, f, indent=4)
    
    return ensemble_results

if __name__ == '__main__':
    import tensorflow as tf
    
    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("\n" + "="*80)
    print("ABLATION STUDY EXPERIMENT")
    print("="*80 + "\n")
    
    # Run ablation study
    ablation_results = run_ablation_study(
        train_dir='../datasets/train/',
        val_dir='../datasets/test/'
    )
    
    print("\n" + "="*80)
    print("TESTING ENSEMBLE")
    print("="*80 + "\n")
    
    # Test ensemble with available models
    model_paths = [
        'models/densenet121_final.keras',
        'models/resnet50_final.keras',
        'models/efficientnetb0_final.keras'
    ]
    
    ensemble_results = test_ensemble(
        model_paths=model_paths,
        val_dir='../datasets/test/'
    )
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE!")
    print("="*80 + "\n")