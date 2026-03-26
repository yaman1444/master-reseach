import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
ax.axis('tight')
ax.axis('off')

# Data for BI-RADS table
col_labels = ['Catégorie BI-RADS', 'Évaluation', 'Probabilité de Malignité']
table_vals = [
    ['0', 'Incomplète –\nNécessite imagerie additionnelle', 'N/A'],
    ['1', 'Négative', 'Essentiellement 0%'],
    ['2', 'Céphalique (Bénigne)', 'Essentiellement 0%'],
    ['3', 'Probablement Bénigne', '>0% mais ≤2%'],
    ['4', 'Anomalie Suspecte', '>2% mais <95%'],
    ['5', 'Hautement Suspecte\nde Malignité', '≥95%'],
    ['6', 'Malignité Prouvée\npar Biopsie', '100%']
]

colors = [['#E3F2FD', '#E3F2FD', '#E3F2FD'],
          ['#C8E6C9', '#C8E6C9', '#C8E6C9'],
          ['#A5D6A7', '#A5D6A7', '#A5D6A7'],
          ['#FFF59D', '#FFF59D', '#FFF59D'],
          ['#FFCC80', '#FFCC80', '#FFCC80'],
          ['#EF9A9A', '#EF9A9A', '#EF9A9A'],
          ['#EF5350', '#EF5350', '#EF5350']]

table = ax.table(cellText=table_vals, colLabels=col_labels, colWidths=[0.2, 0.45, 0.35], cellColours=colors, loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2.5)

# Style headers
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#1565C0')

plt.title('Classification BI-RADS (Breast Imaging Reporting and Data System)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()

os.makedirs('/Users/yaman/master-reseach/memoir', exist_ok=True)
plt.savefig('/Users/yaman/master-reseach/memoir/1_3_birads_microcalcifications.jpg', bbox_inches='tight')
print("✅ BI-RADS Schema generated")
