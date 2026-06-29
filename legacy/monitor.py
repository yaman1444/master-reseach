"""
Moniteur en temps réel pour suivre l'entraînement
Affiche les métriques et détecte les problèmes
"""
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

def monitor_training():
    """Monitore l'entraînement en temps réel"""
    
    print("="*80)
    print("MONITEUR D'ENTRAÎNEMENT - Temps Réel")
    print("="*80)
    print("\nAppuyez sur Ctrl+C pour arrêter le monitoring\n")
    
    models_dir = Path('./models')
    results_dir = Path('./results')
    
    last_check = {}
    start_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Temps écoulé: {timedelta(seconds=int(elapsed))}")
            print("-" * 80)
            
            # Vérifier les modèles sauvegardés
            model_files = list(models_dir.glob('*.keras'))
            if model_files:
                print(f"\n📦 Modèles sauvegardés: {len(model_files)}")
                for model_file in sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                    mtime = model_file.stat().st_mtime
                    age = current_time - mtime
                    size_mb = model_file.stat().st_size / (1024 * 1024)
                    
                    if model_file.name not in last_check or last_check[model_file.name] != mtime:
                        status = "🆕 NOUVEAU"
                        last_check[model_file.name] = mtime
                    else:
                        status = f"   ({int(age)}s ago)"
                    
                    print(f"   {status} {model_file.name} ({size_mb:.1f} MB)")
            
            # Vérifier les résultats
            result_files = list(results_dir.glob('*_results.json'))
            if result_files:
                print(f"\n📊 Résultats disponibles: {len(result_files)}")
                for result_file in sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                            acc = data.get('test_accuracy', 0)
                            f1 = data.get('macro_f1', 0)
                            
                            status = "✅" if acc > 0.90 else "⚠️" if acc > 0.85 else "❌"
                            print(f"   {status} {result_file.name}")
                            print(f"      Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")
                    except:
                        pass
            
            # Vérifier les graphiques
            plot_files = list(results_dir.glob('*_history.png'))
            if plot_files:
                print(f"\n📈 Graphiques: {len(plot_files)}")
                for plot_file in sorted(plot_files, key=lambda x: x.stat().st_mtime, reverse=True)[:2]:
                    mtime = plot_file.stat().st_mtime
                    age = current_time - mtime
                    print(f"   {plot_file.name} ({int(age)}s ago)")
            
            # Détection de problèmes
            print("\n🔍 Diagnostic:")
            
            # Vérifier si entraînement actif
            recent_models = [f for f in model_files if (current_time - f.stat().st_mtime) < 300]
            if recent_models:
                print("   ✅ Entraînement actif (modèle sauvegardé récemment)")
            elif model_files:
                oldest_age = min([current_time - f.stat().st_mtime for f in model_files])
                if oldest_age > 600:
                    print(f"   ⚠️  Aucun modèle sauvegardé depuis {int(oldest_age/60)} minutes")
                    print("      → Vérifier si l'entraînement est bloqué")
            else:
                print("   ⏳ Aucun modèle sauvegardé encore")
            
            # Vérifier performances
            if result_files:
                latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
                with open(latest_result, 'r') as f:
                    data = json.load(f)
                    acc = data.get('test_accuracy', 0)
                    
                    if acc < 0.80:
                        print("   ❌ Accuracy très faible (<80%)")
                        print("      → Vérifier dataset avec diagnose_data.py")
                    elif acc < 0.85:
                        print("   ⚠️  Accuracy sous-optimale (<85%)")
                        print("      → Considérer augmenter learning rate")
                    elif acc < 0.90:
                        print("   ⚠️  Accuracy correcte mais peut être améliorée")
                    else:
                        print("   ✅ Excellentes performances!")
            
            print("\n" + "="*80)
            
            # Attendre 30 secondes
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n✋ Monitoring arrêté par l'utilisateur")
        print("="*80 + "\n")

def show_summary():
    """Affiche un résumé rapide"""
    
    print("\n" + "="*80)
    print("RÉSUMÉ RAPIDE")
    print("="*80 + "\n")
    
    results_dir = Path('./results')
    result_files = list(results_dir.glob('*_results.json'))
    
    if not result_files:
        print("❌ Aucun résultat disponible")
        print("   Exécutez d'abord un entraînement:\n")
        print("   python train_optimized.py")
        return
    
    print("📊 Résultats disponibles:\n")
    
    results = []
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                results.append({
                    'name': result_file.stem.replace('_results', ''),
                    'accuracy': data.get('test_accuracy', 0),
                    'macro_f1': data.get('macro_f1', 0),
                    'time': result_file.stat().st_mtime
                })
        except:
            pass
    
    # Trier par accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    for i, r in enumerate(results, 1):
        status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{status} {r['name']}")
        print(f"   Accuracy: {r['accuracy']:.4f}")
        print(f"   Macro-F1: {r['macro_f1']:.4f}")
        print()
    
    print("="*80 + "\n")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        show_summary()
    else:
        monitor_training()
