# Mathematical Formulas Reference
## Advanced Breast Cancer Classification System

---

## 1. Focal Loss (Lin et al., ICCV'17)

### Standard Cross-Entropy Loss
```
CE(p_t) = -log(p_t)
```
where `p_t = p` if y=1, else `p_t = 1-p`

### Focal Loss Formula
```
FL(p_t) = -α_t(1 - p_t)^γ * log(p_t)
```

**Parameters:**
- `γ ≥ 0` : Focusing parameter (γ=2 in our implementation)
- `α_t ∈ [0,1]` : Class balancing weight (α=0.25)

**Intuition:**
- When `p_t → 1` (easy example): `(1-p_t)^γ → 0`, loss down-weighted
- When `p_t → 0` (hard example): `(1-p_t)^γ → 1`, loss preserved
- Focuses training on hard misclassified examples

**Implementation:**
```python
focal_weight = α * (1 - y_pred)^γ
focal_loss = focal_weight * cross_entropy
```

---

## 2. Mixup Augmentation (Zhang et al., ICLR'18)

### Mixup Formula
```
x̃ = λx_i + (1-λ)x_j
ỹ = λy_i + (1-λ)y_j
```

where:
- `λ ~ Beta(α, α)` with α=0.2
- `(x_i, y_i)` and `(x_j, y_j)` are random training samples

**Beta Distribution:**
```
Beta(α, α) with α=0.2 → λ concentrated near 0 and 1
```

**Effect:**
- Encourages linear behavior between training examples
- Reduces memorization and overfitting
- Smooths decision boundaries

---

## 3. CutMix (Yun et al., ICCV'19)

### CutMix Formula
```
x̃ = M ⊙ x_i + (1-M) ⊙ x_j
ỹ = λy_i + (1-λ)y_j
```

where:
- `M ∈ {0,1}^(H×W)` : Binary mask
- `λ = 1 - (area of cut region / total area)`

**Box Coordinates:**
```
r_w = W * √(1-λ)
r_h = H * √(1-λ)
r_x ~ Uniform(0, W)
r_y ~ Uniform(0, H)
```

---

## 4. CBAM Attention (Woo et al., ECCV'18)

### Channel Attention Module
```
M_c(F) = σ(MLP(AvgPool(F)) + MLP(MaxPool(F)))
```

where:
- `F ∈ R^(H×W×C)` : Input feature map
- `AvgPool, MaxPool : R^(H×W×C) → R^C`
- `MLP : R^C → R^(C/r) → R^C` (r=8, reduction ratio)
- `σ` : Sigmoid activation

**Output:**
```
F' = M_c(F) ⊗ F
```

### Spatial Attention Module
```
M_s(F) = σ(Conv^(7×7)([AvgPool(F); MaxPool(F)]))
```

where:
- `AvgPool, MaxPool : R^(H×W×C) → R^(H×W×1)` (channel-wise)
- `[·;·]` : Concatenation
- `Conv^(7×7)` : 7×7 convolution

**Output:**
```
F'' = M_s(F') ⊗ F'
```

### Complete CBAM
```
F_out = M_s(M_c(F) ⊗ F) ⊗ M_c(F) ⊗ F
```

---

## 5. Grad-CAM (Selvaraju et al., ICCV'17)

### Gradient Computation
```
α^c_k = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_ij)
```

where:
- `y^c` : Score for class c (before softmax)
- `A^k ∈ R^(H×W)` : k-th feature map of last conv layer
- `Z = H × W` : Normalization factor
- `α^c_k` : Importance weight of feature map k for class c

### Grad-CAM Heatmap
```
L^c_Grad-CAM = ReLU(Σ_k α^c_k A_k)
```

**Intuition:**
- Positive gradients → regions that increase class score
- ReLU removes negative influences
- Weighted combination highlights discriminative regions

**Normalization:**
```
L^c_normalized = L^c_Grad-CAM / max(L^c_Grad-CAM)
```

---

## 6. Cosine Annealing Learning Rate

### Formula
```
η_t = η_min + (η_max - η_min) * 0.5 * (1 + cos(πt/T))
```

where:
- `η_t` : Learning rate at epoch t
- `η_max` : Initial learning rate (1e-4 or 1e-5)
- `η_min` : Minimum learning rate (η_max * 0.01)
- `T` : Total number of epochs
- `t` : Current epoch

**Behavior:**
- Starts at `η_max`
- Smoothly decreases following cosine curve
- Reaches `η_min` at epoch T
- Helps escape local minima

---

## 7. Class-Balanced Weights

### Balanced Weight Formula
```
w_i = n_samples / (n_classes * n_samples_i)
```

where:
- `w_i` : Weight for class i
- `n_samples` : Total number of samples
- `n_classes` : Number of classes (3)
- `n_samples_i` : Number of samples in class i

**Example (BUSI dataset):**
```
Benign: 437 samples → w = 780/(3*437) = 0.595
Malignant: 210 samples → w = 780/(3*210) = 1.238
Normal: 133 samples → w = 780/(3*133) = 1.955
```

---

## 8. Macro-F1 Score

### Precision and Recall
```
Precision_i = TP_i / (TP_i + FP_i)
Recall_i = TP_i / (TP_i + FN_i)
```

### F1-Score per Class
```
F1_i = 2 * (Precision_i * Recall_i) / (Precision_i + Recall_i)
```

### Macro-F1 (Unweighted Average)
```
Macro-F1 = (1/n_classes) * Σ_i F1_i
```

**Why Macro-F1?**
- Treats all classes equally (important for imbalanced datasets)
- Penalizes poor performance on minority classes
- Better metric than accuracy for medical diagnosis

---

## 9. AUC-ROC (Area Under Curve)

### ROC Curve
```
TPR(t) = TP(t) / (TP(t) + FN(t))  [True Positive Rate]
FPR(t) = FP(t) / (FP(t) + TN(t))  [False Positive Rate]
```

where `t` is the classification threshold

### AUC Calculation
```
AUC = ∫_0^1 TPR(FPR^(-1)(x)) dx
```

**Interpretation:**
- AUC = 1.0 : Perfect classifier
- AUC = 0.5 : Random classifier
- AUC > 0.9 : Excellent performance

---

## 10. t-SNE Embedding (van der Maaten & Hinton, 2008)

### Cost Function
```
C = Σ_i Σ_j p_ij * log(p_ij / q_ij)
```

where:
- `p_ij` : Similarity in high-dimensional space
- `q_ij` : Similarity in low-dimensional space

### Gaussian Similarity (High-D)
```
p_j|i = exp(-||x_i - x_j||^2 / 2σ_i^2) / Σ_(k≠i) exp(-||x_i - x_k||^2 / 2σ_i^2)
p_ij = (p_j|i + p_i|j) / 2n
```

### Student-t Similarity (Low-D)
```
q_ij = (1 + ||y_i - y_j||^2)^(-1) / Σ_(k≠l) (1 + ||y_k - y_l||^2)^(-1)
```

**Parameters:**
- Perplexity = 30 (effective number of neighbors)
- Iterations = 1000

---

## 11. SHAP Values (Lundberg & Lee, NeurIPS'17)

### Shapley Value Formula
```
φ_i = Σ_(S⊆F\{i}) [|S|!(|F|-|S|-1)! / |F|!] * [f(S∪{i}) - f(S)]
```

where:
- `φ_i` : SHAP value for feature i
- `F` : Set of all features
- `S` : Subset of features
- `f(S)` : Model prediction with feature subset S

**Interpretation:**
- `φ_i > 0` : Feature i increases prediction
- `φ_i < 0` : Feature i decreases prediction
- `|φ_i|` : Magnitude of feature importance

---

## 12. Ensemble Soft Voting

### Soft Voting Formula
```
ŷ = argmax_c Σ_(m=1)^M w_m * p_m(c|x)
```

where:
- `M` : Number of models (3 in our case)
- `w_m` : Weight for model m (w_m = 1/M for equal weights)
- `p_m(c|x)` : Probability of class c from model m

**Our Implementation:**
```
P_ensemble(c|x) = (1/3) * [P_DenseNet(c|x) + P_ResNet(c|x) + P_EfficientNet(c|x)]
```

---

## 13. Progressive Fine-Tuning Strategy

### Phase 1: Feature Extraction
```
θ_base : frozen
θ_head : trainable
L = FocalLoss(f(x; θ_base, θ_head), y)
θ_head ← θ_head - η_1 * ∇_θ_head L
```

### Phase 2: Fine-Tuning
```
θ_base[0:80%] : frozen
θ_base[80%:100%] : trainable
θ_head : trainable
L = FocalLoss(f(x; θ_base, θ_head), y)
θ_trainable ← θ_trainable - η_2 * ∇_θ_trainable L
```

where:
- `η_1 = 1e-4` (Phase 1 learning rate)
- `η_2 = 1e-5` (Phase 2 learning rate, 10× smaller)

---

## 14. CLAHE (Contrast Limited Adaptive Histogram Equalization)

### Histogram Equalization
```
h(v) = round((cdf(v) - cdf_min) / (M*N - cdf_min) * (L-1))
```

where:
- `cdf(v)` : Cumulative distribution function at intensity v
- `M×N` : Image dimensions
- `L` : Number of intensity levels (256)

### Clip Limit
```
clip_limit = (M*N / n_bins) * (1 + α/100)
```

where:
- `n_bins` : Number of histogram bins
- `α` : Clip limit parameter (200 in our case)

**Effect:**
- Enhances local contrast in medical images
- Prevents over-amplification of noise
- Improves visibility of subtle lesions

---

## References

1. Lin, T. Y., et al. (2017). "Focal loss for dense object detection." ICCV.
2. Zhang, H., et al. (2018). "mixup: Beyond empirical risk minimization." ICLR.
3. Yun, S., et al. (2019). "CutMix: Regularization strategy to train strong classifiers." ICCV.
4. Woo, S., et al. (2018). "CBAM: Convolutional block attention module." ECCV.
5. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual explanations from deep networks." ICCV.
6. Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." NeurIPS.
7. van der Maaten, L., & Hinton, G. (2008). "Visualizing data using t-SNE." JMLR.

---

**Note:** All formulas are implemented in the corresponding Python scripts with numerical stability considerations (epsilon values, gradient clipping, etc.).
