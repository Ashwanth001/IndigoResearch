"""
Fix mojibake characters stored literally inside both evaluation notebooks.

The UTF-8 box-drawing/punctuation bytes were incorrectly decoded as cp1252
at some point and then saved back, so now the file contains the literal
mojibake character sequences as Unicode.

Direct replacement map covers all observed bad sequences.
"""
import json

NOTEBOOKS = ['internal_benchmarking.ipynb', 'full_universe_eval.ipynb']

# Map: mojibake string -> correct Unicode character
REPLACEMENTS = [
    ('â€"',  '—'),   # U+2014 em dash
    ('â€"',  '—'),   # alternate encoding artifact
    ('â"€',  '─'),   # U+2500 box-drawing horizontal
    ('â"‚',  '│'),   # U+2502 box-drawing vertical
    ('â–²',  '▲'),   # U+25B2 black up-pointing triangle
    ('â–¼',  '▼'),   # U+25BC black down-pointing triangle
    ('â€™',  '’'),  # right single quotation mark
    ('â€œ',  '“'),  # left double quotation mark
    ('â€\x9d', '”'), # right double quotation mark
    ('Â·',   '·'),   # middle dot
    ('â‰¥',  '≥'),   # greater-than-or-equal
    ('â‰¤',  '≤'),   # less-than-or-equal
    ('â€²',  '′'),   # prime
]

def fix_str(s):
    for bad, good in REPLACEMENTS:
        s = s.replace(bad, good)
    return s

def fix_source(source):
    if isinstance(source, list):
        return [fix_str(line) for line in source]
    return fix_str(source)

for nb_path in NOTEBOOKS:
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)

    fixed_cells = 0
    for cell in nb['cells']:
        original = repr(cell['source'])
        cell['source'] = fix_source(cell['source'])
        if repr(cell['source']) != original:
            fixed_cells += 1

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f'{nb_path}: fixed {fixed_cells} cells')

print('Done.')
