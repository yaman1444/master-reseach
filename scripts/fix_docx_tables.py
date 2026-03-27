import sys
from docx import Document

from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_table_borders(table):
    tbl = table._tbl
    tblPr = tbl.xpath('w:tblPr')
    if not tblPr:
        tblPr = OxmlElement('w:tblPr')
        tbl.insert(0, tblPr)
    else:
        tblPr = tblPr[0]
        
    tblBorders = OxmlElement('w:tblBorders')
    
    for tag in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        edge = OxmlElement(f'w:{tag}')
        edge.set(qn('w:val'), 'single')
        edge.set(qn('w:sz'), '4') # border size
        edge.set(qn('w:space'), '0')
        edge.set(qn('w:color'), '000000') # black
        tblBorders.append(edge)
        
    tblPr.append(tblBorders)

def fix_tables(docx_path):
    print(f"Fixing tables in {docx_path} manually...")
    doc = Document(docx_path)
    
    tables_fixed = 0
    for table in doc.tables:
        set_table_borders(table)
        tables_fixed += 1
    
    doc.save(docx_path)
    print(f"Done! Fixed {tables_fixed} tables with manual borders.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_docx_tables.py <path_to_docx>")
    else:
        fix_tables(sys.argv[1])
