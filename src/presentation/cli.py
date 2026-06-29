import argparse
from src.presentation.config import CONFIG, BASE_DIR
from src.adapters.keras_data_loader import KerasDataLoader
from src.adapters.densenet_adapter import DenseNetAdapter
from src.adapters.efficientnet_adapter import EfficientNetAdapter
from src.adapters.unet_adapter import UNetAdapter
from src.application.train_use_case import TrainUseCase
from src.application.diagnose_use_case import DiagnoseUseCase
from src.application.mask_dataset_use_case import MaskDatasetUseCase

def main():
    parser = argparse.ArgumentParser(description="Hexagonal ML Framework for Breast Cancer Classification")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Run the optimized training pipeline")
    train_parser.add_argument("--model", type=str, choices=["densenet", "efficientnet"], default="densenet", help="Architecture to train")
    train_parser.add_argument("--dataset", type=str, choices=["default", "masked"], default="default", help="Dataset to use")
    
    # Diagnose command
    diagnose_parser = subparsers.add_parser("diagnose", help="Check dataset integrity and structure")
    
    # Mask Data command
    mask_parser = subparsers.add_parser("mask_data", help="Generate masked dataset using U-Net")
    
    args = parser.parse_args()
    
    if args.command == "train":
        if args.dataset == "masked":
            CONFIG['data_dir'] = str(BASE_DIR / 'datasets_masked')
            
        data_loader = KerasDataLoader()
        
        if args.model == "efficientnet":
            model_builder = EfficientNetAdapter()
        else:
            model_builder = DenseNetAdapter()
            
        use_case = TrainUseCase(data_loader, model_builder)
        use_case.execute()
        
    elif args.command == "diagnose":
        use_case = DiagnoseUseCase(CONFIG['data_dir'])
        use_case.execute()
        
    elif args.command == "mask_data":
        unet = UNetAdapter()
        use_case = MaskDatasetUseCase(unet)
        use_case.execute()
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
