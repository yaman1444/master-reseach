import argparse
from src.presentation.config import CONFIG
from src.adapters.keras_data_loader import KerasDataLoader
from src.adapters.densenet_adapter import DenseNetAdapter
from src.application.train_use_case import TrainUseCase
from src.application.diagnose_use_case import DiagnoseUseCase

def main():
    parser = argparse.ArgumentParser(description="Hexagonal ML Framework for Breast Cancer Classification")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Run the optimized training pipeline")
    
    # Diagnose command
    diagnose_parser = subparsers.add_parser("diagnose", help="Check dataset integrity and structure")
    
    args = parser.parse_args()
    
    if args.command == "train":
        data_loader = KerasDataLoader()
        model_builder = DenseNetAdapter()
        use_case = TrainUseCase(data_loader, model_builder)
        use_case.execute()
        
    elif args.command == "diagnose":
        use_case = DiagnoseUseCase(CONFIG['data_dir'])
        use_case.execute()
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
