"""
Fix mojibake in both evaluation notebooks.

Observed bad sequences (codepoints identified from raw inspection):
  â€" = [0xe2, 0x20ac, 0x201d] -> — (em dash U+2014)
  â"€ = [0xe2, 0x201d, 0x20ac] -> ─ (box drawing horizontal U+2500)
  â†'  = [0xe2, 0x2020, 0x2019] -> → (rightwards arrow U+2192) [in comments]

The triple encoding happened because:
  Original UTF-8 bytes (e.g. E2 80 94 for em dash)
  were misread as Windows-1252 / latin-1 giving multi-char sequences,
  then those chars were each stored as their own Unicode codepoints.
"""
import re, sys
sys.stdout.reconfigure(encoding='utf-8')

NOTEBOOKS = ['internal_benchmarking.ipynb', 'full_universe_eval.ipynb']

# Key: the literal unicode string in the file -> replacement
# Codepoints confirmed by direct inspection
REPLACEMENTS = [
    ('\xe2€”',  '—'),  # â€" -> — (em dash)
    ('\xe2”€',  '─'),  # â"€ -> ─ (box horizontal)
    ('\xe2†’',  '→'),  # â†' -> → (arrow)
    ('\xe2€™',  '’'),  # â€™ -> ' (right single quote)
    ('\xe2€œ',  '“'),  # â€œ -> " (left double quote)
    ('\xe2€\x9d',    '”'),  # â€\x9d -> " (right double quote)
    ('\xc2\xb7',          '·'),  # Â· -> · (middle dot)
    ('\xc2\xa0',          ' '),  # Â  -> non-breaking space
    ('\xe2‰\xa4',    '≤'),  # â‰¤ -> ≤
    ('\xe2‰\xa5',    '≥'),  # â‰¥ -> ≥
]

for nb_path in NOTEBOOKS:
    with open(nb_path, encoding='utf-8') as f:
        raw = f.read()

    original = raw
    for bad, good in REPLACEMENTS:
        raw = raw.replace(bad, good)

    replaced = sum(original.count(bad) for bad, _ in REPLACEMENTS)

    with open(nb_path, 'w', encoding='utf-8') as f:
        f.write(raw)

    print(f'{nb_path}: replaced {replaced} occurrences across all patterns')

# Verify
for nb_path in NOTEBOOKS:
    with open(nb_path, encoding='utf-8') as f:
        raw = f.read()
    bad = re.findall(r'\xe2.{0,3}', raw)
    print(f'{nb_path}: remaining bad sequences = {len(bad)}')
    for b in sorted(set(bad))[:5]:
        print(f'  {repr(b)} -> {[hex(ord(c)) for c in b]}')

print('Done.')
