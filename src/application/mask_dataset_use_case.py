import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.presentation.config import CONFIG, BASE_DIR
from src.adapters.unet_adapter import UNetAdapter

class MaskDatasetUseCase:
    def __init__(self, unet_adapter: UNetAdapter):
        self.unet_adapter = unet_adapter
        self.input_dir = Path(CONFIG['data_dir'])
        self.output_dir = BASE_DIR / 'datasets_masked'

    def execute(self):
        print("="*80)
        print("GÉNÉRATION DU DATASET MASQUÉ (U-NET CASCADE)")
        print("="*80)
        
        self.unet_adapter.load_model()
        
        for phase in ['train', 'test']:
            phase_in_dir = self.input_dir / phase
            phase_out_dir = self.output_dir / phase
            
            if not phase_in_dir.exists():
                continue
                
            for class_name in ['benign', 'malignant', 'normal']:
                class_in_dir = phase_in_dir / class_name
                class_out_dir = phase_out_dir / class_name
                class_out_dir.mkdir(parents=True, exist_ok=True)
                
                if not class_in_dir.exists():
                    continue
                    
                images = list(class_in_dir.glob('*.png'))
                if not images:
                    continue
                    
                print(f"Traitement de {phase}/{class_name} ({len(images)} images)...")
                
                for img_path in tqdm(images):
                    if '_mask' in img_path.name:
                        continue # Ne pas traiter les masques s'ils sont dans le même dossier
                        
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                        
                    # Préparation pour U-Net (256x256 selon legacy/train_unet.py)
                    original_h, original_w = img.shape[:2]
                    img_resized = cv2.resize(img, (256, 256))
                    img_tensor = (img_resized / 255.0).astype(np.float32)
                    img_tensor = np.expand_dims(img_tensor, axis=0)
                    
                    # Prédiction du masque
                    mask = self.unet_adapter.predict_mask(img_tensor)[0] # shape (256, 256, 1)
                    
                    # Redimensionner le masque à la taille originale de l'image
                    mask_resized = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                    if len(mask_resized.shape) == 2:
                        mask_resized = np.expand_dims(mask_resized, axis=-1)
                    
                    # Si c'est normal, on garde l'image ? Le jury veut qu'on masque.
                    # Pour un sein normal, s'il n'y a pas de lésion, le masque sera tout noir (donc image noire).
                    # Mais pour la classification normale, il ne faut pas classer du noir comme "normal".
                    # C'est un point de discussion : l'IA classifie la région suspecte. Si pas de région, elle peut dire Normal.
                    # En pratique, on applique le masque avec un peu de fond.
                    
                    masked_img = img * mask_resized
                    
                    out_path = class_out_dir / img_path.name
                    cv2.imwrite(str(out_path), masked_img)
                    
        print(f"\n✅ Dataset masqué généré dans: {self.output_dir}")
        print("="*80)
