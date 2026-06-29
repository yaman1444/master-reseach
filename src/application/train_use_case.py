import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pathlib import Path

from src.presentation.config import CONFIG, MODELS_DIR, RESULTS_DIR
from src.ports.data_port import DataPort
from src.ports.model_port import ModelPort
from src.domain.focal_loss import FocalLoss

class TrainUseCase:
    def __init__(self, data_loader: DataPort, model_builder: ModelPort):
        self.data_loader = data_loader
        self.model_builder = model_builder

    def execute(self):
        print("="*80)
        print("ENTRAÎNEMENT DENSENET121 - ARCHITECTURE HEXAGONALE")
        print("="*80)
        
        train_gen, test_gen, class_weights = self.data_loader.get_train_test_generators(
            CONFIG['data_dir'],
            CONFIG['img_size'],
            CONFIG['batch_size']
        )
        
        model, base_model = self.model_builder.build_model(
            CONFIG['img_size'],
            CONFIG['dropout_rate'],
            CONFIG['l2_reg']
        )
        
        # PHASE 1
        print("\n" + "="*80)
        print("PHASE 1: Training head (base frozen)")
        print("="*80 + "\n")
        
        model.compile(
            optimizer=keras.optimizers.Adam(CONFIG['initial_lr']),
            loss=FocalLoss(gamma=2.0, alpha=0.25),
            metrics=['accuracy']
        )
        
        callbacks_p1 = [
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, mode='max', restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, mode='max', min_lr=1e-6, verbose=1),
            keras.callbacks.ModelCheckpoint(str(MODELS_DIR / 'densenet121_phase1.keras'), monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
        ]
        
        history1 = model.fit(
            train_gen,
            epochs=CONFIG['initial_epochs'],
            validation_data=test_gen,
            class_weight=class_weights,
            callbacks=callbacks_p1,
            verbose=1
        )
        
        # PHASE 2
        print("\n" + "="*80)
        print("PHASE 2: Fine-tuning (top 20% unfrozen)")
        print("="*80 + "\n")
        
        base_model.trainable = True
        total_layers = len(base_model.layers)
        freeze_until = int(total_layers * 0.8)
        
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
        
        trainable = sum([1 for l in base_model.layers if l.trainable])
        print(f"Unfrozen layers: {trainable}/{total_layers}\n")
        
        model.compile(
            optimizer=keras.optimizers.Adam(CONFIG['fine_tune_lr']),
            loss=FocalLoss(gamma=2.0, alpha=0.25),
            metrics=['accuracy']
        )
        
        callbacks_p2 = [
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, mode='max', restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=4, mode='max', min_lr=1e-7, verbose=1),
            keras.callbacks.ModelCheckpoint(str(MODELS_DIR / 'densenet121_optimized.keras'), monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
        ]
        
        history2 = model.fit(
            train_gen,
            epochs=CONFIG['fine_tune_epochs'],
            validation_data=test_gen,
            class_weight=class_weights,
            callbacks=callbacks_p2,
            verbose=1
        )
        
        self._evaluate_and_save(model, test_gen, history1, history2)
        
    def _evaluate_and_save(self, model, test_gen, history1, history2):
        print("\n" + "="*80)
        print("FINAL EVALUATION")
        print("="*80 + "\n")
        
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
        
        y_pred_probs = model.predict(test_gen, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_gen.classes
        
        metrics = {}
        for i, class_name in enumerate(['benign', 'malignant', 'normal']):
            tp = np.sum((y_true == i) & (y_pred == i))
            fp = np.sum((y_true != i) & (y_pred == i))
            fn = np.sum((y_true == i) & (y_pred != i))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[class_name] = {'precision': precision, 'recall': recall, 'f1': f1}
        
        macro_f1 = np.mean([m['f1'] for m in metrics.values()])
        
        print(f"📊 Final Results:")
        print(f"   Accuracy:  {test_acc:.4f}")
        print(f"   Macro-F1:  {macro_f1:.4f}\n")
        
        results = {
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'macro_f1': float(macro_f1),
            'metrics_by_class': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in metrics.items()},
            'config': CONFIG
        }
        
        with open(RESULTS_DIR / 'densenet121_optimized_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        self._plot_history(history1, history2)
        print(f"\n✅ Training Completed. Models and Results saved.")
        print("="*80 + "\n")
        
    def _plot_history(self, history1, history2):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        acc = history1.history['accuracy'] + history2.history['accuracy']
        val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
        loss = history1.history['loss'] + history2.history['loss']
        val_loss = history1.history['val_loss'] + history2.history['val_loss']
        
        epochs = range(1, len(acc) + 1)
        phase1_end = len(history1.history['accuracy'])
        
        axes[0].plot(epochs, acc, 'b-', label='Train')
        axes[0].plot(epochs, val_acc, 'r-', label='Validation')
        axes[0].axvline(x=phase1_end, color='g', linestyle='--')
        axes[0].set_title('Accuracy')
        axes[0].legend()
        
        axes[1].plot(epochs, loss, 'b-', label='Train')
        axes[1].plot(epochs, val_loss, 'r-', label='Validation')
        axes[1].axvline(x=phase1_end, color='g', linestyle='--')
        axes[1].set_title('Loss')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'densenet121_optimized_history.png', dpi=150, bbox_inches='tight')
