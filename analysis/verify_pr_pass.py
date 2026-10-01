import io
import re
import sys

import pymupdf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

doc = pymupdf.open('paper_PR/paper_PR_submission.pdf')
txt = '\n'.join(p.get_text() for p in doc)
doc.close()
norm = re.sub(r'\s+', ' ', txt)

CHECKS = {
    'refuted regulariser wording': 'implicit regulariser',
    'retracted 90% reduction': '90% reduction',
    'retracted three of eight': 'three of eight metrics',
    'three apparently significant bins': 'three apparently significant bins',
    'stop after S4': 'stop after',
    '5.4 six same-seed runs': 'six same-seed runs',
    '5.4 six distinct weight digests': 'six distinct weight digests',
    '5.4 scope boundary': 'aggregate-metric level only',
    '5.4 pycocotools bin boundary': 'every size bin but all undefined',
    '5.4 DINO argued': 'argued rather than measured',
    '5.4 jitter property of operator': 'property of the operator, not of one architecture',
    '6.3 transient 1.18-1.21': '1.18\u20131.21 per step',
    '6.3 d=0.219 / 0.205 at step 100': 'd = 0.219 after 100 steps',
    '6.3 factor 1.000 (0.998-1.002)': '1.000 (0.998\u20131.002',
    '6.3 real-activation noise': '3.8\u00d710\u22128\u20138.4\u00d710\u22128',
    '6.3 synthetic noise': '1.2\u00d710\u22129 to 9.9\u00d710\u22129',
    '6.3 three- to eight-fold': 'three- to eight-fold',
    '6.3 order of magnitude below': 'order of magnitude below',
    '6.3 conclusion 1': 'the chain now closes on the real model',
    '6.3 conclusion 2': 'basin distance, not a rate',
    'float Table 1 caption': 'VisDrone noise floor over the 15 runs',
    'float Table 2 caption': 'Divergence of the full detector',
    'float Figure 1 caption': 'injected perturbation of the measured hardware-noise',
    'float Algorithm 1': 'Gather-based bilinear sampling',
    'ratio range 2.2 to 10.2': '2.2 (the',
    'p > 0.05 every metric': 'p > 0.05 on every metric',
    'sampler-only sigma 0.00299 n=3': '0.00299',
    'paired t=1.31 p=0.32': 't = 1.31, p = 0.32',
    'bad-run denominator 8': '1/8',
    'corrected F df (5, 9)': '(5, 9)',
    'survey macros': '2.76',
}
bad = []
for name, needle in CHECKS.items():
    ok = needle in norm
    print(('OK   ' if ok else 'MISS ') + name + ('' if ok else '  <= ' + repr(needle)))
    if not ok:
        bad.append(name)
print()
print('all present' if not bad else 'MISSING: ' + ', '.join(bad))
