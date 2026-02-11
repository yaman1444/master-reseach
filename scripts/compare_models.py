"""
Multi-Model Comparison: DenseNet121 vs ResNet50 vs EfficientNetB0
Trains all models on same BUSI split and generates comparison table
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from train_advanced import train_model, CONFIG

def compare_models(train_dir='../datasets/train/', val_dir='../datasets/test/'):
    """Train and compare multiple architectures"""
    
    models_to_compare = ['densenet121', 'resnet50', 'efficientnetb0']
    all_results = {}
    
    for model_name in models_to_compare:
        print(f"\n{'='*80}")
        print(f"TRAINING: {model_name.upper()}")
        print(f"{'='*80}\n")
        
        try:
            model, results, history = train_model(
                model_name=model_name,
                train_dir=train_dir,
                val_dir=val_dir,
                config=CONFIG
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
    
    # Generate comparison table
    generate_comparison_table(all_results)
    plot_model_comparison(all_results)
    
    return all_results

def generate_comparison_table(all_results):
    """
    Generate markdown table with metrics per model and class
    
    Columns: Model | Accuracy | Macro-F1 | Normal-F1 | Benign-F1 | Malignant-F1 | AUC-ROC
    """
    rows = []
    
    for model_name, results in all_results.items():
        report = results['classification_report']
        auc_scores = results['auc_scores']
        
        # Get class names (assuming order: benign, malignant, normal)
        class_names = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        row = {
            'Model': model_name.upper(),
            'Accuracy': f"{results['accuracy']:.4f}",
            'Macro-F1': f"{results['macro_f1']:.4f}",
        }
        
        # Add per-class F1 scores
        for class_name in sorted(class_names):
            if class_name in report:
                row[f'{class_name.capitalize()}-F1'] = f"{report[class_name]['f1-score']:.4f}"
                row[f'{class_name.capitalize()}-Precision'] = f"{report[class_name]['precision']:.4f}"
                row[f'{class_name.capitalize()}-Recall'] = f"{report[class_name]['recall']:.4f}"
        
        # Add AUC-ROC
        if isinstance(auc_scores, list) and len(auc_scores) > 0:
            row['Mean-AUC'] = f"{np.mean(auc_scores):.4f}"
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save as CSV
    df.to_csv('results/model_comparison.csv', index=False)
    
    # Print markdown table
    print("\n" + "="*100)
    print("MODEL COMPARISON TABLE")
    print("="*100 + "\n")
    print(df.to_markdown(index=False))
    
    # Save markdown
    with open('results/model_comparison.md', 'w') as f:
        f.write("# Model Comparison Results\n\n")
        f.write("## Overall Performance\n\n")
        f.write(df[['Model', 'Accuracy', 'Macro-F1', 'Mean-AUC']].to_markdown(index=False))
        f.write("\n\n## Per-Class Performance\n\n")
        f.write(df.to_markdown(index=False))
    
    return df

def plot_model_comparison(all_results):
    """Plot comparison charts for all models"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = list(all_results.keys())
    
    # 1. Overall Accuracy and Macro-F1
    ax = axes[0, 0]
    accuracies = [all_results[m]['accuracy'] for m in models]
    macro_f1s = [all_results[m]['macro_f1'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x + width/2, macro_f1s, width, label='Macro-F1', alpha=0.8)
    ax.set_ylabel('Score')
    ax.set_title('Overall Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.8, 1.0])
    
    # Add value labels on bars
    for i, (acc, f1) in enumerate(zip(accuracies, macro_f1s)):
        ax.text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Per-class F1 scores
    ax = axes[0, 1]
    
    # Extract class names from first model
    first_model = list(all_results.values())[0]
    class_names = [k for k in first_model['classification_report'].keys() 
                   if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    for i, model_name in enumerate(models):
        f1_scores = []
        for class_name in class_names:
            report = all_results[model_name]['classification_report']
            if class_name in report:
                f1_scores.append(report[class_name]['f1-score'])
            else:
                f1_scores.append(0)
        
        ax.bar(x + i*width, f1_scores, width, label=model_name.upper(), alpha=0.8)
    
    ax.set_ylabel('F1-Score')
    ax.set_title('Per-Class F1-Score Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.capitalize() for c in class_names], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.7, 1.0])
    
    # 3. AUC-ROC comparison
    ax = axes[1, 0]
    
    auc_means = []
    for model_name in models:
        auc_scores = all_results[model_name]['auc_scores']
        if isinstance(auc_scores, list) and len(auc_scores) > 0:
            auc_means.append(np.mean(auc_scores))
        else:
            auc_means.append(0)
    
    bars = ax.bar(models, auc_means, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Mean AUC-ROC')
    ax.set_title('Mean AUC-ROC Comparison')
    ax.set_xticklabels([m.upper() for m in models], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.8, 1.0])
    
    # Add value labels
    for bar, auc in zip(bars, auc_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Confusion matrices comparison (heatmap grid)
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create mini confusion matrices
    n_models = len(models)
    mini_fig, mini_axes = plt.subplots(1, n_models, figsize=(12, 3))
    
    for idx, model_name in enumerate(models):
        cm = np.array(all_results[model_name]['confusion_matrix'])
        
        if n_models == 1:
            mini_ax = mini_axes
        else:
            mini_ax = mini_axes[idx]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   cbar=False, ax=mini_ax, square=True)
        mini_ax.set_title(model_name.upper(), fontsize=10)
        mini_ax.set_xlabel('Predicted', fontsize=8)
        mini_ax.set_ylabel('True', fontsize=8)
        mini_ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    mini_fig.savefig('results/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(mini_fig)
    
    # Main figure
    plt.tight_layout()
    fig.savefig('results/models_comparison_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nComparison plots saved to results/")

if __name__ == '__main__':
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "="*80)
    print("MULTI-MODEL COMPARISON EXPERIMENT")
    print("Models: DenseNet121, ResNet50, EfficientNetB0")
    print("="*80 + "\n")
    
    results = compare_models(
        train_dir='../datasets/train/',
        val_dir='../datasets/test/'
    )
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("Results saved to results/model_comparison.csv and .md")
    print("="*80 + "\n")
