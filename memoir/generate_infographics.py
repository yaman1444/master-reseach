import matplotlib.pyplot as plt
import numpy as np
import os

# Set global aesthetics (Premium Look)
plt.style.use('ggplot')

out_dir = "/Users/yaman/master-reseach/memoir"

# 1. 1_1_epidemiologie.jpg
def plot_epidemiology():
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    labels = ['Cancer du sein\n(3 869 cas)', 'Autres cancers\nféminins']
    sizes = [33.5, 66.5]
    colors = ['#E91E63', '#CFD8DC']
    explode = (0.05, 0)
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=140, textprops={'fontsize': 14, 'weight': 'bold'})
           
    for autotext in autotexts:
        autotext.set_color('white' if autotext.get_text().startswith('33.5') else 'black')
        
    ax.axis('equal')
    plt.title("Répartition des cancers féminins\nen Côte d'Ivoire (GLOBOCAN 2022)", fontsize=16, weight='bold', pad=20, color='#333333')
    plt.savefig(os.path.join(out_dir, '1_1_epidemiologie.jpg'), bbox_inches='tight')
    plt.close()

# 2. 1_2_infrastructures.jpg
def plot_infrastructure():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    countries = ['France', "Côte d'Ivoire"]
    ratio = [10, 1] # Pour 1 million de femmes
    colors = ['#90CAF9', '#FFAB91']
    
    bars = ax.barh(countries, ratio, color=colors, height=0.6)
    ax.set_xlabel('Nombre estimé de mammographes pour 1 million de femmes', fontsize=12, weight='bold')
    ax.set_title("Disponibilité des infrastructures de dépistage (Ordre de grandeur)", fontsize=16, weight='bold', pad=20, color='#333333')
    
    for bar in bars:
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                str(int(bar.get_width())), 
                color='black', va='center', fontsize=14, weight='bold')
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '1_2_infrastructures.jpg'), bbox_inches='tight')
    plt.close()

# 3. 3_1_dataset_split.jpg
def plot_dataset_split():
    labels = ['Entraînement\n[Augmenté hors-ligne]', 'Validation\n[15%]', 'Test\n[15%]']
    sizes = [70, 15, 15]
    colors = ['#1976D2', '#FFC107', '#4CAF50']
    explode = (0.05, 0, 0)
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.0f%%',
           startangle=90, textprops={'fontsize': 14, 'weight': 'bold'}, wedgeprops=dict(width=0.4, edgecolor='w', linewidth=2))
    
    for autotext in autotexts:
        autotext.set_color('white')
        
    ax.axis('equal')
    plt.title("Répartition stricte des données\n(Sans fuite de données)", fontsize=16, weight='bold', pad=20, color='#333333')
    plt.savefig(os.path.join(out_dir, '3_1_dataset_split.jpg'), bbox_inches='tight')
    plt.close()

# 4. 3_2_class_imbalance.jpg
def plot_class_imbalance():
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    classes = ['Normal', 'Début/Suspect\n(Bénin)', 'Grave\n(Malin)']
    counts = [133, 437, 210]
    colors = ['#4CAF50', '#FF9800', '#E53935']
    
    bars = ax.bar(classes, counts, color=colors, width=0.5)
    ax.set_ylabel("Nombre d'images", fontsize=12, weight='bold')
    ax.set_title('Déséquilibre des classes dans le dataset original (BUSI)', fontsize=16, weight='bold', pad=20, color='#333333')
    
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(int(bar.get_height())), 
                color='black', ha='center', fontsize=12, weight='bold')
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '3_2_class_imbalance.jpg'), bbox_inches='tight')
    plt.close()

# 5. 4_6_recall_comparison.jpg
def plot_recall_comparison():
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    models = ['Standard (Exp 3)', 'Calibré Fail-Safe (Exp 5)']
    recalls = [71.9, 90.4]
    
    bars = ax.bar(models, recalls, color=['#9E9E9E', '#E91E63'], width=0.4)
    ax.set_ylabel('Recall (%)', fontsize=14, weight='bold')
    ax.set_title("Évolution du Recall sur la classe suspecte 'Début'", fontsize=16, weight='bold', pad=20, color='#333333')
    ax.set_ylim(0, 105)
    
    # Target line
    ax.axhline(90, color='black', linestyle='--', linewidth=2, label='Cible clinique (90%)')
    ax.legend(fontsize=12)
    
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f"{bar.get_height()}%", 
                color='black', ha='center', fontsize=14, weight='bold')
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '4_6_recall_comparison.jpg'), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Génération des graphiques...")
    plot_epidemiology()
    plot_infrastructure()
    plot_dataset_split()
    plot_class_imbalance()
    plot_recall_comparison()
    print("Succès ! Images générées dans :", out_dir)
