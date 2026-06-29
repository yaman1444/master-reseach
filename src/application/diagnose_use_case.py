import os
from pathlib import Path

class DiagnoseUseCase:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def execute(self):
        print("="*80)
        print("DIAGNOSTIC DU DATASET")
        print("="*80)
        
        if not self.data_dir.exists():
            print(f"❌ ERROR: Dataset folder not found at {self.data_dir}")
            return
            
        train_dir = self.data_dir / 'train'
        test_dir = self.data_dir / 'test'
        
        for phase, phase_dir in [("Train", train_dir), ("Test", test_dir)]:
            if not phase_dir.exists():
                print(f"❌ ERROR: Phase folder missing: {phase_dir}")
                continue
                
            print(f"\n📁 Phase: {phase}")
            for class_name in ['debut', 'grave', 'normal']:
                class_dir = phase_dir / class_name
                if class_dir.exists():
                    count = len(list(class_dir.glob('*.png')))
                    print(f"   - {class_name:10s}: {count} images")
                else:
                    print(f"   - {class_name:10s}: 0 images (FOLDER MISSING)")
                    
        print("\n✅ Diagnostic completed.")
        print("="*80)
