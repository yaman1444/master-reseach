# Document d'Exigences - Système Avancé de Classification du Cancer du Sein par Deep Learning

## Introduction

Ce document définit les exigences pour un système de recherche avancé de classification du cancer du sein utilisant l'apprentissage profond sur des images échographiques. Le projet vise à transformer un prototype existant (84,3% de précision, 82,3% macro-F1) en un pipeline de recherche de qualité publication pour les revues Medical Image Analysis ou IEEE TMI.

Le système actuel utilise DenseNet121 avec attention CBAM sur le dataset BUSI (780 images : 437 bénignes, 210 malignes, 133 normales). L'objectif ambitieux est d'atteindre >96% macro-F1 tout en maintenant une rigueur scientifique et une explicabilité clinique.

## Glossaire

- **System**: Le système complet de classification du cancer du sein par Deep Learning
- **Training_Pipeline**: Le pipeline d'entraînement incluant prétraitement, augmentation, et optimisation
- **Model_Architecture**: L'architecture du réseau de neurones (DenseNet121, Vision Transformer, etc.)
- **XAI_Module**: Le module d'explicabilité (Grad-CAM, SHAP, attention maps)
- **Validation_Framework**: Le framework de validation statistique et cross-dataset
- **Augmentation_Engine**: Le moteur d'augmentation de données médicales
- **Ensemble_System**: Le système d'ensemble de modèles
- **Calibration_Module**: Le module de calibration des seuils et confiances
- **Metrics_Reporter**: Le système de rapport des métriques cliniques
- **Experiment_Tracker**: Le système de suivi des expériences et reproductibilité
- **BUSI_Dataset**: Breast Ultrasound Images Dataset (780 images, 3 classes)
- **Macro_F1**: Moyenne non pondérée des F1-scores par classe
- **Recall_Malignant**: Sensibilité pour la classe maligne (priorité clinique)
- **Statistical_Validator**: Le module de validation statistique (intervalles de confiance, tests)

## Exigences

### Exigence 1 : Architecture de Modèle Avancée

**User Story:** En tant que chercheur en IA médicale, je veux explorer des architectures de pointe au-delà de DenseNet121, afin d'améliorer les performances tout en maintenant l'explicabilité.

#### Critères d'Acceptation

1. THE Model_Architecture SHALL support Vision Transformers (ViT, Swin Transformer)
2. THE Model_Architecture SHALL support EfficientNet variants (B0 through B7)
3. THE Model_Architecture SHALL support hybrid CNN-Transformer architectures
4. WHEN a new architecture is evaluated, THE System SHALL compare it against DenseNet121 baseline using identical training conditions
5. THE Model_Architecture SHALL preserve attention mechanism capabilities for explainability
6. FOR ALL architectures tested, THE Experiment_Tracker SHALL record hyperparameters, training time, and memory usage
7. THE Model_Architecture SHALL support transfer learning from ImageNet and medical imaging pretrained weights

### Exigence 2 : Augmentation de Données Médicales Spécialisée

**User Story:** En tant que chercheur, je veux des stratégies d'augmentation adaptées aux images médicales, afin de maximiser l'efficacité des données sur le petit dataset BUSI.

#### Critères d'Acceptation

1. THE Augmentation_Engine SHALL implement medical-specific augmentations beyond CLAHE and Mixup
2. THE Augmentation_Engine SHALL support elastic deformations simulating tissue variations
3. THE Augmentation_Engine SHALL support speckle noise addition characteristic of ultrasound imaging
4. THE Augmentation_Engine SHALL support CutMix augmentation strategy
5. THE Augmentation_Engine SHALL support AutoAugment or RandAugment with medical imaging search space
6. WHEN augmentation is applied, THE Augmentation_Engine SHALL preserve anatomical plausibility
7. THE Augmentation_Engine SHALL support test-time augmentation (TTA) for inference
8. FOR ALL augmentation strategies, THE System SHALL conduct ablation studies measuring individual contribution to performance

### Exigence 3 : Gestion Avancée du Déséquilibre de Classes

**User Story:** En tant que chercheur, je veux des techniques avancées pour gérer le déséquilibre 3-classes (437/210/133), afin d'optimiser les performances sur toutes les classes.

#### Critères d'Acceptation

1. THE Training_Pipeline SHALL support class-balanced sampling strategies
2. THE Training_Pipeline SHALL support focal loss with configurable gamma and alpha parameters
3. THE Training_Pipeline SHALL support class-weighted loss functions
4. THE Training_Pipeline SHALL support SMOTE or ADASYN oversampling techniques adapted for images
5. THE Training_Pipeline SHALL support two-stage training (balance then fine-tune)
6. WHEN class imbalance techniques are applied, THE Metrics_Reporter SHALL report per-class metrics separately
7. THE System SHALL compare at least 3 different imbalance handling strategies with statistical validation

### Exigence 4 : Fonctions de Perte Avancées

**User Story:** En tant que chercheur, je veux explorer des fonctions de perte au-delà de Focal Loss, afin d'optimiser spécifiquement pour les métriques cliniques.

#### Critères d'Acceptation

1. THE Training_Pipeline SHALL support Dice Loss for segmentation-inspired classification
2. THE Training_Pipeline SHALL support Tversky Loss with configurable false positive/negative weights
3. THE Training_Pipeline SHALL support compound loss functions (e.g., Focal + Dice)
4. THE Training_Pipeline SHALL support asymmetric loss functions prioritizing recall for malignant class
5. THE Training_Pipeline SHALL support label smoothing techniques
6. FOR ALL loss functions, THE System SHALL conduct ablation studies measuring impact on Recall_Malignant
7. WHEN a loss function is selected, THE Training_Pipeline SHALL allow hyperparameter tuning via grid search or Bayesian optimization

### Exigence 5 : Explicabilité et Interprétabilité Avancées (XAI)

**User Story:** En tant que chercheur visant l'adoption clinique, je veux des outils d'explicabilité robustes, afin de générer la confiance des cliniciens et analyser les cas d'échec.

#### Critères d'Acceptation

1. THE XAI_Module SHALL implement Grad-CAM++ for improved localization over Grad-CAM
2. THE XAI_Module SHALL implement Integrated Gradients for attribution analysis
3. THE XAI_Module SHALL implement attention map visualization for transformer-based models
4. THE XAI_Module SHALL generate saliency maps overlaid on original ultrasound images
5. WHEN a prediction is made, THE XAI_Module SHALL provide confidence scores with uncertainty quantification
6. THE XAI_Module SHALL support failure case analysis identifying systematic error patterns
7. THE XAI_Module SHALL generate comparative visualizations between correct and incorrect predictions
8. FOR ALL XAI methods, THE System SHALL validate that highlighted regions align with clinical tumor locations

### Exigence 6 : Validation Croisée et Robustesse Multi-Dataset

**User Story:** En tant que chercheur, je veux valider la généralisation du modèle sur des datasets externes, afin de démontrer la robustesse clinique.

#### Critères d'Acceptation

1. THE Validation_Framework SHALL implement stratified k-fold cross-validation (k=5 minimum)
2. THE Validation_Framework SHALL support external validation on CBIS-DDSM dataset
3. THE Validation_Framework SHALL support domain adaptation techniques for cross-dataset transfer
4. WHEN external validation is performed, THE Validation_Framework SHALL report performance degradation metrics
5. THE Validation_Framework SHALL implement confidence calibration using temperature scaling or Platt scaling
6. THE Validation_Framework SHALL generate calibration curves (reliability diagrams)
7. THE Validation_Framework SHALL test robustness to image quality variations (blur, noise, contrast)
8. FOR ALL validation experiments, THE Statistical_Validator SHALL compute 95% confidence intervals using bootstrap resampling

### Exigence 7 : Métriques Cliniques et Validation Statistique

**User Story:** En tant que chercheur visant une publication, je veux des métriques cliniques rigoureuses avec validation statistique, afin de respecter les standards des revues médicales.

#### Critères d'Acceptation

1. THE Metrics_Reporter SHALL report sensitivity (recall) and specificity for each class
2. THE Metrics_Reporter SHALL report positive predictive value (PPV) and negative predictive value (NPV)
3. THE Metrics_Reporter SHALL compute macro-F1, weighted-F1, and per-class F1 scores
4. THE Metrics_Reporter SHALL generate ROC curves with AUC for each class (one-vs-rest)
5. THE Metrics_Reporter SHALL generate precision-recall curves with average precision scores
6. THE Statistical_Validator SHALL compute 95% confidence intervals for all metrics using bootstrap (n=1000 iterations minimum)
7. THE Statistical_Validator SHALL perform McNemar's test for comparing paired model predictions
8. THE Statistical_Validator SHALL perform DeLong's test for comparing AUC scores
9. WHEN Recall_Malignant is reported, THE Metrics_Reporter SHALL prioritize this metric as primary clinical outcome
10. THE Metrics_Reporter SHALL generate confusion matrices with absolute counts and percentages
11. FOR ALL experiments, THE Statistical_Validator SHALL test statistical significance at p<0.05 threshold

### Exigence 8 : Méthodes d'Ensemble Avancées

**User Story:** En tant que chercheur, je veux des stratégies d'ensemble sophistiquées, afin de maximiser les performances en combinant plusieurs modèles.

#### Critères d'Acceptation

1. THE Ensemble_System SHALL support majority voting across multiple architectures
2. THE Ensemble_System SHALL support weighted voting based on validation performance
3. THE Ensemble_System SHALL support stacking with meta-learner (logistic regression, XGBoost)
4. THE Ensemble_System SHALL support snapshot ensembling from single training run
5. THE Ensemble_System SHALL support test-time augmentation ensembling
6. WHEN ensemble predictions are made, THE Ensemble_System SHALL provide uncertainty estimates via prediction variance
7. THE Ensemble_System SHALL compare ensemble performance against best single model with statistical tests
8. THE Ensemble_System SHALL analyze diversity metrics between ensemble members (disagreement, Q-statistic)

### Exigence 9 : Reproductibilité et Suivi des Expériences

**User Story:** En tant que chercheur, je veux une reproductibilité complète de toutes les expériences, afin de respecter les standards scientifiques et faciliter la révision par les pairs.

#### Critères d'Acceptation

1. THE Experiment_Tracker SHALL record all random seeds (Python, NumPy, PyTorch, CUDA)
2. THE Experiment_Tracker SHALL record exact package versions (requirements.txt with pinned versions)
3. THE Experiment_Tracker SHALL record all hyperparameters for each experiment
4. THE Experiment_Tracker SHALL record training curves (loss, accuracy, F1) at each epoch
5. THE Experiment_Tracker SHALL record GPU memory usage and training time
6. THE Experiment_Tracker SHALL version control all model checkpoints with metadata
7. THE Experiment_Tracker SHALL integrate with MLflow or Weights & Biases for experiment tracking
8. WHEN an experiment is completed, THE Experiment_Tracker SHALL generate a reproducibility report with all configuration details
9. THE Experiment_Tracker SHALL support experiment comparison dashboards
10. FOR ALL experiments, THE System SHALL save predictions on test set for post-hoc analysis

### Exigence 10 : Optimisation des Hyperparamètres

**User Story:** En tant que chercheur, je veux une recherche systématique d'hyperparamètres, afin d'optimiser les performances de manière rigoureuse.

#### Critères d'Acceptation

1. THE Training_Pipeline SHALL support grid search for hyperparameter optimization
2. THE Training_Pipeline SHALL support random search for hyperparameter optimization
3. THE Training_Pipeline SHALL support Bayesian optimization (Optuna or similar)
4. THE Training_Pipeline SHALL optimize learning rate, batch size, weight decay, and dropout rate
5. THE Training_Pipeline SHALL optimize augmentation hyperparameters
6. THE Training_Pipeline SHALL optimize loss function hyperparameters (focal gamma, Tversky alpha/beta)
7. WHEN hyperparameter search is performed, THE System SHALL use nested cross-validation to avoid overfitting
8. THE Experiment_Tracker SHALL record all hyperparameter trials with corresponding validation performance
9. THE System SHALL visualize hyperparameter importance using partial dependence plots

### Exigence 11 : Analyse des Cas d'Échec et Diagnostic d'Erreurs

**User Story:** En tant que chercheur, je veux une analyse systématique des erreurs du modèle, afin d'identifier les limitations et orienter les améliorations futures.

#### Critères d'Acceptation

1. THE System SHALL identify and catalog all misclassified images from validation and test sets
2. THE System SHALL compute error rates stratified by true class and predicted class
3. THE XAI_Module SHALL generate explanation visualizations for all misclassified cases
4. THE System SHALL cluster misclassified images to identify systematic error patterns
5. THE System SHALL analyze correlation between prediction confidence and correctness
6. THE System SHALL identify ambiguous cases where multiple models disagree
7. WHEN failure analysis is performed, THE System SHALL generate a report with representative error examples
8. THE System SHALL compare error patterns across different model architectures

### Exigence 12 : Comparaison avec l'État de l'Art

**User Story:** En tant que chercheur visant une publication, je veux comparer mon système avec les méthodes de l'état de l'art, afin de démontrer la contribution scientifique.

#### Critères d'Acceptation

1. THE System SHALL implement at least 3 baseline methods from recent literature (2020-2024)
2. THE System SHALL reproduce published results on BUSI dataset when possible
3. THE System SHALL compare against classical machine learning baselines (SVM, Random Forest with handcrafted features)
4. WHEN comparisons are made, THE System SHALL use identical train/test splits across all methods
5. THE Statistical_Validator SHALL test statistical significance of improvements over baselines
6. THE System SHALL generate comparison tables with all metrics for publication
7. THE System SHALL document any differences in experimental setup from published baselines

### Exigence 13 : Calibration de Confiance et Seuils Optimaux

**User Story:** En tant que chercheur, je veux optimiser les seuils de décision et calibrer les scores de confiance, afin de maximiser les métriques cliniques prioritaires.

#### Critères d'Acceptation

1. THE Calibration_Module SHALL optimize classification thresholds to maximize Recall_Malignant while maintaining minimum precision
2. THE Calibration_Module SHALL implement temperature scaling for confidence calibration
3. THE Calibration_Module SHALL implement isotonic regression for calibration
4. THE Calibration_Module SHALL generate reliability diagrams before and after calibration
5. THE Calibration_Module SHALL compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
6. WHEN threshold optimization is performed, THE Calibration_Module SHALL use validation set only (not test set)
7. THE Calibration_Module SHALL support multi-threshold optimization for different clinical scenarios (screening vs diagnosis)
8. THE Metrics_Reporter SHALL report performance at both default (0.5) and optimized thresholds

### Exigence 14 : Documentation de Qualité Publication

**User Story:** En tant que chercheur, je veux une documentation complète prête pour publication, afin de faciliter la soumission aux revues et la révision par les pairs.

#### Critères d'Acceptation

1. THE System SHALL generate LaTeX tables for all experimental results
2. THE System SHALL generate high-resolution figures (300 DPI minimum) for all visualizations
3. THE System SHALL generate a methods section describing the complete pipeline
4. THE System SHALL generate a results section with statistical comparisons
5. THE System SHALL maintain a bibliography of all referenced methods and datasets
6. THE System SHALL generate supplementary materials with detailed ablation studies
7. WHEN documentation is generated, THE System SHALL follow Medical Image Analysis or IEEE TMI formatting guidelines
8. THE System SHALL generate a reproducibility checklist per journal requirements

### Exigence 15 : Parsing et Sérialisation des Configurations Expérimentales

**User Story:** En tant que chercheur, je veux charger et sauvegarder les configurations expérimentales de manière fiable, afin de garantir la reproductibilité et faciliter le partage.

#### Critères d'Acceptation

1. WHEN a valid experiment configuration file is provided, THE Parser SHALL parse it into an ExperimentConfig object
2. WHEN an invalid configuration file is provided, THE Parser SHALL return a descriptive error indicating the problematic field
3. THE Pretty_Printer SHALL format ExperimentConfig objects back into valid YAML or JSON configuration files
4. FOR ALL valid ExperimentConfig objects, parsing then printing then parsing SHALL produce an equivalent object (round-trip property)
5. THE Parser SHALL validate that all required fields are present (model_name, dataset_path, hyperparameters)
6. THE Parser SHALL validate that hyperparameter values are within acceptable ranges
7. THE Pretty_Printer SHALL include comments documenting each configuration parameter

### Exigence 16 : Pipeline d'Entraînement Progressif et Fine-Tuning

**User Story:** En tant que chercheur, je veux un pipeline d'entraînement progressif, afin d'optimiser le transfert learning sur le petit dataset BUSI.

#### Critères d'Acceptation

1. THE Training_Pipeline SHALL support frozen backbone training followed by full fine-tuning
2. THE Training_Pipeline SHALL support layer-wise learning rate decay (discriminative learning rates)
3. THE Training_Pipeline SHALL support gradual unfreezing of layers
4. THE Training_Pipeline SHALL support cosine annealing learning rate schedule
5. THE Training_Pipeline SHALL support warm restarts (SGDR)
6. WHEN progressive training is used, THE Training_Pipeline SHALL log performance after each training stage
7. THE Training_Pipeline SHALL support early stopping based on validation Recall_Malignant
8. THE Training_Pipeline SHALL save checkpoints at best validation performance and at regular intervals

### Exigence 17 : Gestion des Données et Prétraitement

**User Story:** En tant que chercheur, je veux un pipeline de données robuste, afin de garantir la qualité et la cohérence du prétraitement.

#### Critères d'Acceptation

1. THE System SHALL load BUSI_Dataset with stratified train/validation/test splits (70/15/15)
2. THE System SHALL normalize images using dataset statistics (mean, std)
3. THE System SHALL resize images to configurable input dimensions while preserving aspect ratio
4. THE System SHALL support CLAHE preprocessing for contrast enhancement
5. THE System SHALL validate data integrity (no corrupted images, correct labels)
6. WHEN data is loaded, THE System SHALL report class distribution for each split
7. THE System SHALL support data caching for faster loading in subsequent epochs
8. THE System SHALL handle missing or corrupted mask files gracefully

### Exigence 18 : Métriques d'Invariance et Propriétés Métamorphiques

**User Story:** En tant que chercheur, je veux tester les propriétés d'invariance du modèle, afin de valider la robustesse aux transformations attendues.

#### Critères d'Acceptation

1. WHEN an image is horizontally flipped, THE Model_Architecture SHALL produce predictions with similar confidence (within 5% tolerance)
2. WHEN an image is rotated by small angles (-10° to +10°), THE Model_Architecture SHALL maintain the same predicted class
3. WHEN brightness is adjusted within clinical range, THE Model_Architecture SHALL maintain prediction consistency
4. FOR ALL augmentation transformations that preserve diagnosis, THE System SHALL verify prediction stability
5. THE Validation_Framework SHALL report invariance violation rate across test set
6. THE System SHALL identify images where invariance properties are violated for further analysis

### Exigence 19 : Tests de Propriétés Basés sur des Modèles Simplifiés

**User Story:** En tant que chercheur, je veux comparer les prédictions avec des modèles simplifiés, afin de valider que la complexité apporte une valeur ajoutée.

#### Critères d'Acceptation

1. THE System SHALL implement a simple baseline (e.g., ResNet18 or MobileNet)
2. WHEN the advanced model and simple baseline disagree, THE System SHALL analyze which is correct more often
3. THE System SHALL measure the performance gap between complex and simple models
4. THE Statistical_Validator SHALL test if the performance improvement justifies the complexity increase
5. THE System SHALL analyze computational cost vs performance trade-offs

### Exigence 20 : Validation des Conditions d'Erreur et Robustesse

**User Story:** En tant que chercheur, je veux tester le comportement du système face à des entrées invalides ou dégradées, afin de garantir la robustesse clinique.

#### Critères d'Acceptation

1. WHEN an image with extreme noise is provided, THE System SHALL either reject it or provide low confidence predictions
2. WHEN an image outside the expected size range is provided, THE System SHALL handle it gracefully with appropriate preprocessing
3. WHEN an image with artifacts (e.g., text overlays, measurement markers) is provided, THE System SHALL detect and flag potential quality issues
4. THE System SHALL implement input validation rejecting non-medical images
5. THE System SHALL provide uncertainty estimates that increase for out-of-distribution inputs
6. WHEN corrupted or incomplete images are encountered, THE System SHALL log errors without crashing
7. THE Validation_Framework SHALL test robustness using adversarial perturbations (FGSM, PGD) to measure model vulnerability

