# scan_patterns.py - scan a file for risk-control trigger PATTERNS (URLs,
# emails, phone-like digits, long digit runs) WITHOUT revealing content.
# Prints only pattern type + byte position + ASCII token list.
import os
import re
import sys

def scan(path):
    b = open(path, 'rb').read()
    t = b.decode('utf-8', errors='replace')
    n_ascii = sum(1 for c in t if ord(c) < 128)
    n_cjk = sum(1 for c in t if '\u4e00' <= c <= '\u9fff')
    pats = {
        'url': r'https?://\S+',
        'www': r'www\.\S+',
        'email': r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+',
        'phone': r'(?:\+?\d[\d\s-]{7,}\d)',
        'id18': r'\d{17}[\dXx]',
        'longdigits': r'\d{6,}',
    }
    found = {}
    for name, p in pats.items():
        m = re.search(p, t)
        found[name] = (1, m.start()) if m else (0, -1)
    tokens = [ch for ch in re.split(r'[^\x20-\x7E]+', t)
              if ch and re.search(r'[A-Za-z0-9]', ch)]
    return b, n_ascii, n_cjk, found, tokens

for p in sys.argv[1:]:
    b, na, nc, found, toks = scan(p)
    stats = ' '.join('%s=%d@%d' % (k, v[0], v[1]) for k, v in found.items())
    label = os.path.basename(os.path.dirname(p))
    print('FILE=%s bytes=%d ascii=%d cjk=%d %s' % (label, len(b), na, nc, stats))
    print('TOKENS[%d]: %s' % (len(toks), ' '.join(toks[:200])))
