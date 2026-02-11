"""
Script principal: Orchestration des optimisations
"""
import sys
import subprocess
from pathlib import Path

def run_script(script_name, description):
    """Exécuter un script Python"""
    print("\n" + "="*80)
    print(f"EXÉCUTION: {description}")
    print("="*80 + "\n")
    
    result = subprocess.run([sys.executable, script_name], cwd='.')
    
    if result.returncode != 0:
        print(f"\n❌ Erreur lors de l'exécution de {script_name}")
        return False
    
    print(f"\n✅ {description} terminé")
    return True

def main():
    print("\n" + "="*80)
    print("PIPELINE D'OPTIMISATION - MOUSSOKENE MASTER SEARCH")
    print("="*80 + "\n")
    
    print("Options disponibles:")
    print("  1. Calibration des seuils (30 min)")
    print("  2. Test calibration sur images")
    print("  3. Amélioration classe normal (1h)")
    print("  4. Ablation study (2-3h)")
    print("  5. K-fold validation (3-4h)")
    print("  6. Résumé des résultats")
    print("  7. Tout exécuter (6-8h)")
    print("  0. Quitter")
    
    choice = input("\nChoisir une option: ")
    
    if choice == '1':
        run_script('calibrate_thresholds.py', 'Calibration des seuils')
    
    elif choice == '2':
        run_script('test_calibration.py', 'Test calibration')
    
    elif choice == '3':
        run_script('improve_normal_class.py', 'Amélioration classe normal')
    
    elif choice == '4':
        run_script('ablation_study_v2.py', 'Ablation study')
    
    elif choice == '5':
        run_script('kfold_validation.py', 'K-fold validation')
    
    elif choice == '6':
        run_script('summary_results.py', 'Résumé des résultats')
    
    elif choice == '7':
        print("\n⚠️  Exécution complète (6-8h)")
        confirm = input("Confirmer? (y/n): ")
        if confirm.lower() == 'y':
            run_script('calibrate_thresholds.py', 'Calibration')
            run_script('test_calibration.py', 'Test calibration')
            run_script('improve_normal_class.py', 'Amélioration normal')
            run_script('ablation_study_v2.py', 'Ablation study')
            run_script('kfold_validation.py', 'K-fold validation')
            run_script('summary_results.py', 'Résumé final')
    
    elif choice == '0':
        print("Au revoir!")
        return
    
    else:
        print("Option invalide")

if __name__ == '__main__':
    main()
