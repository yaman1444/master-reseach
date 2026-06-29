"""
Quick Start Script - Run All Experiments
Executes complete pipeline: training, comparison, ablation, visualizations
"""
import os
import sys
import time

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def run_experiment(script_name, description):
    """Run a Python script and track time"""
    print_section(description)
    start_time = time.time()
    
    try:
        os.system(f'python {script_name}')
        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed in {elapsed/60:.2f} minutes")
        return True
    except Exception as e:
        print(f"\n✗ Error in {description}: {e}")
        return False

def main():
    """Run complete experimental pipeline"""
    
    print_section("BREAST CANCER CLASSIFICATION - COMPLETE PIPELINE")
    print("This will run all experiments sequentially:")
    print("1. Advanced Training (DenseNet121)")
    print("2. Model Comparison (DenseNet/ResNet/EfficientNet)")
    print("3. Ablation Study")
    print("4. Grad-CAM Visualizations")
    print("5. Advanced Visualizations (t-SNE, UMAP, SHAP)")
    print("\nEstimated time: 3-5 hours (with GPU)")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    total_start = time.time()
    results = {}
    
    # Experiment 1: Advanced Training
    results['training'] = run_experiment(
        'train_advanced.py',
        'EXPERIMENT 1: Advanced Training (DenseNet121)'
    )
    
    # Experiment 2: Model Comparison
    if results['training']:
        results['comparison'] = run_experiment(
            'compare_models.py',
            'EXPERIMENT 2: Multi-Model Comparison'
        )
    
    # Experiment 3: Ablation Study
    results['ablation'] = run_experiment(
        'ablation_study.py',
        'EXPERIMENT 3: Ablation Study'
    )
    
    # Experiment 4: Grad-CAM
    if results['training']:
        results['gradcam'] = run_experiment(
            'visualize_gradcam.py',
            'EXPERIMENT 4: Grad-CAM Visualizations'
        )
    
    # Experiment 5: Advanced Visualizations
    if results['training']:
        results['visualizations'] = run_experiment(
            'visualize_all.py',
            'EXPERIMENT 5: Advanced Visualizations'
        )
    
    # Summary
    total_elapsed = time.time() - total_start
    
    print_section("PIPELINE COMPLETE - SUMMARY")
    print(f"Total execution time: {total_elapsed/3600:.2f} hours\n")
    
    print("Experiment Results:")
    for exp_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {exp_name.capitalize()}: {status}")
    
    print("\n" + "="*80)
    print("OUTPUT FILES:")
    print("="*80)
    
    # List generated files
    if os.path.exists('results'):
        print("\nResults directory:")
        for file in sorted(os.listdir('results')):
            print(f"  - results/{file}")
    
    if os.path.exists('models'):
        print("\nModels directory:")
        for file in sorted(os.listdir('models')):
            if file.endswith('.keras'):
                print(f"  - models/{file}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review results in results/ directory")
    print("2. Check training logs: tensorboard --logdir=logs/")
    print("3. Load best model: tf.keras.models.load_model('models/densenet121_final.keras')")
    print("4. See README.md for detailed analysis")
    print("\n")

if __name__ == '__main__':
    main()