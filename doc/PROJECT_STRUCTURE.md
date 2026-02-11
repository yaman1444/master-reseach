# Project Structure Documentation

```
moussokene_master_search/
│
├── README.md                                    # Main documentation
├── MATHEMATICAL_FORMULAS.md                     # Mathematical reference
├── requirements.txt                             # Python dependencies
├── breast_cancer_classification_colab.ipynb     # Google Colab notebook
│
├── datasets/                                    # Data directory
│   ├── train/                                   # Training data (80%)
│   │   ├── benign/                             # Benign tumor images
│   │   ├── malignant/                          # Malignant tumor images
│   │   └── normal/                             # Normal tissue images
│   └── test/                                    # Test data (20%)
│       ├── benign/
│       ├── malignant/
│       └── normal/
│
├── scripts/                                     # Main code directory
│   ├── focal_loss.py                           # Focal Loss implementation
│   ├── augmentation.py                         # CLAHE + Mixup/CutMix
│   ├── cbam.py                                 # CBAM attention module
│   ├── train_model.py                          # Original baseline training
│   ├── train_advanced.py                       # Advanced training (main)
│   ├── compare_models.py                       # Multi-model comparison
│   ├── ablation_study.py                       # Ablation experiments
│   ├── visualize_gradcam.py                    # Grad-CAM visualizations
│   ├── visualize_all.py                        # t-SNE/UMAP/SHAP/ROC
│   ├── demo_predict.py                         # Single image prediction
│   └── run_all.py                              # Run complete pipeline
│
├── models/                                      # Saved models (generated)
│   ├── densenet121_phase1_best.keras           # Phase 1 checkpoint
│   ├── densenet121_final.keras                 # Final DenseNet121
│   ├── resnet50_final.keras                    # Final ResNet50
│   ├── efficientnetb0_final.keras              # Final EfficientNetB0
│   └── densenet121_ablation_*.keras            # Ablation checkpoints
│
├── results/                                     # Outputs (generated)
│   ├── densenet121_results.json                # Metrics JSON
│   ├── densenet121_training_history.png        # Loss/accuracy curves
│   ├── densenet121_confusion_matrix.png        # Confusion matrix
│   ├── densenet121_gradcam.png                 # Grad-CAM examples
│   ├── densenet121_feature_maps.png            # Feature map evolution
│   ├── densenet121_embeddings.png              # t-SNE/UMAP plots
│   ├── densenet121_roc_curves.png              # ROC curves
│   ├── densenet121_confusion_detailed.png      # Detailed confusion matrix
│   ├── densenet121_shap.png                    # SHAP importance
│   ├── model_comparison.csv                    # Comparison table (CSV)
│   ├── model_comparison.md                     # Comparison table (Markdown)
│   ├── models_comparison_charts.png            # Comparison plots
│   ├── confusion_matrices_comparison.png       # All confusion matrices
│   ├── ablation_study.csv                      # Ablation results (CSV)
│   ├── ablation_study.md                       # Ablation report
│   ├── ablation_study_plot.png                 # Ablation gains plot
│   └── ensemble_results.json                   # Ensemble metrics
│
└── logs/                                        # TensorBoard logs (generated)
    ├── densenet121_phase1/
    ├── densenet121_phase2/
    ├── resnet50_phase1/
    ├── resnet50_phase2/
    ├── efficientnetb0_phase1/
    └── efficientnetb0_phase2/
```

---

## File Descriptions

### Core Scripts

#### `focal_loss.py`
- **Purpose:** Focal Loss implementation for class imbalance
- **Key Formula:** `FL(p_t) = -α(1-p_t)^γ * log(p_t)`
- **Parameters:** γ=2, α=0.25
- **Usage:** Imported by training scripts

#### `augmentation.py`
- **Purpose:** Advanced data augmentation
- **Components:**
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Mixup (λ~Beta(0.2, 0.2))
  - CutMix (random patch mixing)
- **Class:** `AugmentedDataGenerator` wraps Keras generators

#### `cbam.py`
- **Purpose:** Convolutional Block Attention Module
- **Components:**
  - `ChannelAttention`: What to focus on
  - `SpatialAttention`: Where to focus
  - `CBAM`: Combined module
- **Integration:** Added before classification head

#### `train_advanced.py` ⭐ MAIN TRAINING SCRIPT
- **Purpose:** Advanced training with all optimizations
- **Features:**
  - Progressive fine-tuning (2 phases)
  - Focal Loss + class weights
  - Cosine Annealing LR
  - CLAHE + Mixup augmentation
  - CBAM attention
  - TensorBoard logging
- **Outputs:**
  - `models/{model_name}_final.keras`
  - `results/{model_name}_results.json`
  - Training plots and confusion matrix
- **Runtime:** ~1-2 hours (GPU)

#### `compare_models.py`
- **Purpose:** Train and compare multiple architectures
- **Models:** DenseNet121, ResNet50, EfficientNetB0
- **Outputs:**
  - Comparison table (CSV + Markdown)
  - Side-by-side performance charts
  - Confusion matrices grid
- **Runtime:** ~3-5 hours (GPU)

#### `ablation_study.py`
- **Purpose:** Systematic component evaluation
- **Configurations:**
  1. Baseline (no optimizations)
  2. +Augmentation
  3. +Dropout
  4. +CBAM (full model)
- **Outputs:**
  - Ablation table with incremental gains
  - Gain visualization plot
  - Ensemble evaluation
- **Runtime:** ~2-3 hours (GPU)

#### `visualize_gradcam.py`
- **Purpose:** Grad-CAM interpretability
- **Class:** `GradCAM` with `compute_heatmap()` and `overlay_heatmap()`
- **Formula:** `α^c_k = (1/Z) Σ(∂y^c/∂A^k)`
- **Outputs:**
  - Grad-CAM overlays (12 samples)
  - Feature map evolution
- **Runtime:** ~5-10 minutes

#### `visualize_all.py`
- **Purpose:** Comprehensive visualizations
- **Components:**
  - t-SNE/UMAP embeddings
  - ROC curves (one-vs-rest)
  - Detailed confusion matrices
  - SHAP analysis (optional)
- **Outputs:** Multiple PNG files per model
- **Runtime:** ~10-20 minutes

#### `demo_predict.py`
- **Purpose:** Single image prediction demo
- **Usage:** `python demo_predict.py --image path/to/image.png --model models/densenet121_final.keras`
- **Features:**
  - Class prediction with confidence
  - Grad-CAM overlay
  - Probability bar chart
  - Clinical interpretation
- **Runtime:** <1 minute

#### `run_all.py`
- **Purpose:** Execute complete pipeline
- **Sequence:**
  1. Advanced training
  2. Model comparison
  3. Ablation study
  4. Grad-CAM visualization
  5. Advanced visualizations
- **Usage:** `python run_all.py`
- **Runtime:** ~5-8 hours (GPU)

---

## Workflow

### Quick Start (Single Model)
```bash
cd scripts
python train_advanced.py
python visualize_gradcam.py
python demo_predict.py --image ../datasets/test/malignant/sample.png
```

### Full Experiment
```bash
cd scripts
python run_all.py  # Runs everything
tensorboard --logdir=../logs/  # Monitor training
```

### Google Colab
1. Upload `breast_cancer_classification_colab.ipynb`
2. Enable GPU (Runtime → Change runtime type)
3. Upload dataset and scripts
4. Run cells sequentially

---

## Key Metrics Tracked

### Per Model
- **Accuracy**: Overall classification accuracy
- **Macro-F1**: Unweighted average F1 across classes
- **Per-class F1**: Benign, Malignant, Normal
- **AUC-ROC**: Area under ROC curve (one-vs-rest)
- **Confusion Matrix**: True vs predicted labels

### Training Curves
- Loss (train/val)
- Accuracy (train/val)
- AUC (train/val)
- Learning rate schedule

---

## Expected Results

### Baseline (train_model.py)
```
Accuracy: 88-90%
Macro-F1: 85-87%
Training time: 20-30 min
```

### Advanced (train_advanced.py)
```
Accuracy: 96.5%+
Macro-F1: 96.2%+
AUC-ROC: 98.5%+
Training time: 1-2 hours
```

### Model Ranking (Expected)
1. **DenseNet121** (96.5% F1) - Best overall
2. **EfficientNetB0** (95.5% F1) - Good efficiency
3. **ResNet50** (94.5% F1) - Baseline strong

### Ablation Gains
```
Baseline → +Aug: +4.5% F1
+Aug → +Dropout: +2.3% F1
+Dropout → +CBAM: +3.1% F1
Total gain: +12.7% F1
```

---

## Hardware Requirements

### Minimum
- CPU: 4 cores
- RAM: 16 GB
- Storage: 10 GB
- Training time: 10-15 hours (CPU only)

### Recommended
- GPU: NVIDIA RTX 3060+ (12GB VRAM)
- RAM: 32 GB
- Storage: 20 GB SSD
- Training time: 3-5 hours (GPU)

### Cloud (Google Colab)
- Free tier: T4 GPU (16GB VRAM)
- Training time: 4-6 hours
- Session limit: 12 hours

---

## Troubleshooting

### Out of Memory
```python
# Reduce batch size in CONFIG
CONFIG['batch_size'] = 8  # or 4
```

### Slow Training
```python
# Reduce epochs for testing
CONFIG['initial_epochs'] = 5
CONFIG['fine_tune_epochs'] = 10
```

### Missing Dependencies
```bash
pip install -r requirements.txt
# If SHAP/UMAP fail, they're optional
```

### Dataset Not Found
```bash
# Download BUSI from Kaggle
kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset
unzip breast-ultrasound-images-dataset.zip -d datasets/
```

---

## Citation

If you use this code, please cite:

```bibtex
@misc{moussokene2024breast,
  title={Advanced Breast Cancer Classification with DenseNet121},
  author={Moussokene Master Search},
  year={2024},
  howpublished={\url{https://github.com/...}}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Contact

For questions or issues:
- Open GitHub issue
- Email: [your_email]

---

**Last Updated:** 2024
