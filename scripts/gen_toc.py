import re

with open('/Users/yaman/master-reseach/memoir/Memoir_Master_classification_cancer_breast.md', 'r') as f:
    lines = f.readlines()

toc = ["## **TABLE DES MATIÈRES**\n"]

for line in lines:
    if line.startswith('## '):
        t = line.replace('## ', '').replace('**', '').replace('\\', '').strip()
        
        slug = t.lower()
        slug = re.sub(r'[^\w\s\-àáâãäåèéêëìíîïòóôõöùúûüýÿçñ]', '', slug)
        slug = re.sub(r'\s+', '-', slug)
        
        if "CHAPITRE" in t or "CONCLUSION" in t or "ANNEXES" in t or "Annexe" in t or "LISTE" in t:
            toc.append(f"- [{t}](#{slug})")
        elif t.startswith('1.') or t.startswith('2.') or t.startswith('3.') or t.startswith('4.') or t.startswith('5.'):
            parts = t.split(' ')[0].split('.')
            if len(parts) > 2 and parts[2].isdigit():
                toc.append(f"    - [{t}](#{slug})")
            else:
                toc.append(f"  - [{t}](#{slug})")

with open('/Users/yaman/master-reseach/scripts/gen_toc.txt', 'w') as f:
    f.write('\n'.join(toc))
print("TOC written")
