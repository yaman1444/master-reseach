with open('/Users/yaman/master-reseach/memoir/Memoir_Master_classification_cancer_breast.md', 'r') as f:
    lines = f.readlines()

with open('/Users/yaman/master-reseach/scripts/gen_toc.txt', 'r') as f:
    toc_text = f.read()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if "TABLE DES MATIÈRES" in line and i < 100:
        start_idx = i
    if "LISTE DES ABRÉVIATIONS ................................................. 8" in line and i > start_idx and i < 300:
        end_idx = i

if start_idx != -1 and end_idx != -1:
    new_lines = lines[:start_idx] + [toc_text + "\n\n"] + lines[end_idx+2:]
else:
    new_lines = lines
    print("TOC boundaries not found exactly, but continuing to clean headers.")

# Clean headers
for i in range(len(new_lines)):
    if new_lines[i].startswith('## **'):
        # also remove any trailing backslashes or asterisk 
        new_lines[i] = new_lines[i].replace('## **', '## ').replace('**', '').replace('\\', '')
    elif new_lines[i].startswith('## '):
        new_lines[i] = new_lines[i].replace('**', '').replace('\\', '')

with open('/Users/yaman/master-reseach/memoir/Memoir_Master_classification_cancer_breast.md', 'w') as f:
    f.writelines(new_lines)

print("File updated with new TOC and clean headers!")
