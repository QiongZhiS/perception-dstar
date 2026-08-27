# analyze_log_ascii.py
# Structural analysis of logs/transduction_run.log BYTES without revealing
# non-ASCII content. Outputs only ASCII text + [NONASCII xN] placeholders.
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(ROOT, 'logs', 'transduction_run.log')
b = open(P, 'rb').read()
ok_pos = b.find(b'[ok]')
err_pos = b.find(b'[err]')
na = [(i, x) for i, x in enumerate(b) if x >= 128]
skeleton = []
i = 0
while i < len(b):
    if b[i] < 128:
        j = i
        while j < len(b) and b[j] < 128:
            j += 1
        skeleton.append(b[i:j].decode('ascii', errors='replace'))
        i = j
    else:
        j = i
        while j < len(b) and b[j] >= 128:
            j += 1
        skeleton.append('[NONASCII x%d]' % (j - i))
        i = j
print('log_bytes=%d ok_pos=%s err_pos=%s nonascii_count=%d first_na=%s last_na=%s sha256=%s' % (
    len(b), ok_pos, err_pos, len(na), na[0][0] if na else -1, na[-1][0] if na else -1,
    hashlib.sha256(b).hexdigest()[:16]))
print('SKELETON: ' + '|'.join(skeleton))
