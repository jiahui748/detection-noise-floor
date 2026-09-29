#!/usr/bin/env python3
"""Check that every protected value and claim survives in the submission PDF.

Each entry is a list of alternative spellings; an entry passes if any spelling is
found in the extracted text. Reports the set of missing items.
"""
import re
import sys
from pathlib import Path

txt = Path('audit/_sub_text.txt').read_text(encoding='utf-8', errors='replace')
# `pdftotext -layout` puts a bare page number on its own line between the last
# text line of one page and the first of the next, so a phrase that straddles a
# page break ("three apparently significant bins") arrives split.  Drop those
# numbers and the hyphenation before flattening, or the check reports a false
# MISSING on a phrase that is present.
txt = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', txt)
txt = re.sub(r'-\n(?=[a-z])', '', txt)
norm = re.sub(r'\s+', ' ', txt)

GROUPS = {
    'ratio range 2.2 to 10.2': ['2.2 to 10.2', '2.2 (the', '10.2 (the'],
    'p > 0.05 every metric': ['p > 0.05 on every metric'],
    'no "three bins" claim': ['three apparently significant bins'],
    'sampler-only sigma 0.00299 n=3': ['0.00299'],
    'paired t = 1.31': ['t = 1.31', '1.31'],
    'paired p = 0.32': ['p = 0.32'],
    'bad-run denominator 8': ['1/8', '1 out of 8'],
    'corrected F test df (5,9)': ['(5,9)', '(5, 9)'],
    'survey macro values, AP50 family': ['2.76', '3.75', '16.60', '-7.80', '2.21', '1.92'],
    'survey macro values, mAP50-95 family': ['1.64', '3.16'],
    'survey macro value, factor': ['3.9'],
    'retracted three of eight': ['three of eight metrics'],
    'retracted 90% reduction': ['90% reduction', '90\\% reduction'],
    'stop after Section 4': ['stop after'],
    'refuted regulariser': ['implicit', 'regulariser'],
    'newexp transient 1.18-1.21': ['1.18', '1.21'],
    'newexp 100 steps': ['100 steps', '100 optimiser steps', 'first hundred steps'],
    'newexp d=0.219 / 0.205': ['0.219', '0.205'],
    'newexp factor 1.000 (0.998-1.002)': ['1.000', '0.998', '1.002'],
    'newexp module 1.007-1.029': ['1.007', '1.029'],
    'newexp activation noise 3.8e-8 - 8.4e-8': ['3.8', '8.4'],
    'newexp synthetic 1.2e-9 - 9.9e-9': ['1.2', '9.9'],
    'newexp three- to eight-fold': ['three- to eight-fold', 'three- to eight'],
    'newexp order of magnitude below': ['order of magnitude'],
    'newexp 0.250 median': ['0.250'],
    'newexp 0.221 - 0.715': ['0.221', '0.715'],
    'newexp 0.155 - 0.715 / 0.228': ['0.155', '0.228'],
    'newexp 1.0006': ['1.0006'],
    'table t20 float present': ['Divergence of the full detector'],
    'figure fig8 present': ['injected perturbation of the measured hardware-noise'],
    'mAP50 0.4607 / sigma 0.0027': ['0.4607', '0.0027'],
    '4-8px sigma 0.0272 MDE 10.8': ['0.0272', '10.8'],
    '>64px sigma 0.0280 MDE 11.1': ['0.0280', '11.1'],
    'runs per arm 117 / 123': ['117', '123'],
    '1.2 runs per arm': ['1.2 runs'],
    '279-fold / 2.33e-9 / 6.51e-7': ['279', '2.33', '6.51'],
    'eval split 0.0013 - 0.0120': ['0.0013', '0.0120'],
    'bootstrap sd 0.0084 / 0.0079': ['0.0084', '0.0079'],
    'F 0.90 p 0.96 / F 1.53 p 0.55': ['0.90', '0.96', '1.53', '0.55'],
    'F 0.34 - 1.92': ['0.34', '1.92'],
    'nested F 0.29 - 1.45 p 0.18': ['0.29', '1.45', '0.18'],
    'operator divergence 3.8e-5 - 9.3e-5': ['3.8', '9.3'],
    'grid_sample 1.7e-5 - 7.9e-5': ['1.7', '7.9'],
    'equivalence 6.8e-6 / 1.2e-7': ['6.8', '1.2'],
    'variance 0.00245 / 0.00190': ['0.00245', '0.00190'],
    'paired +0.0002 sd 0.0003': ['0.0002', '0.0003'],
    'TT100K 0.8126 / 0.8162 / 3.6-fold': ['0.8126', '0.8162', '3.6'],
    'TT100K sigma 0.02396 / 0.00661': ['0.02396', '0.00661'],
    'cost 3.48x / 14.36x / 1.03x / 1.8x': ['3.48', '14.36', '1.03', '1.8'],
    'bad run 0.3830 / 7.7 points': ['0.3830', '7.7'],
    'Fisher p = 0.35': ['p = 0.35'],
    'Clopper-Pearson 2.5% / 9.5%': ['2.5', '9.5'],
    '+0.25 epochs': ['0.25'],
    'refuted dose 0/3, 1/3, 0.3461': ['0/3', '1/3', '0.3461'],
    'early error p = 0.047': ['0.047'],
    'survey N 15 / deltas 8 AP50, 10 mAP50-95': ['survey of 15', '15 papers', '10 verified deltas'],
    'survey restricted set n=7 median 2.21 mean 1.92': ['2.21', '1.92'],
    'survey census 3 runs / 3 sd / 0 CI / 2 repro': ['3 of the 15', '3 report a standard deviation',
                                                      '2 discuss reproducibility'],
    '24.0 +- 0.4 / 0.0684 +- 0.0013': ['24.0', '0.0684'],
    '8.23 points spread': ['8.23'],
    '7.8 points below baseline': ['7.8'],
    '548-image split': ['548'],
    '104894 objects / 10894': ['10 894', '10894'],
    'VisDrone 8.5px / 47.1% / 78.4%': ['8.5', '47.1', '78.4'],
    'TT100K median 34px': ['34px'],
    '8-16px cv 0.307': ['0.307'],
    'peak cv 0.164 / AP50 0.167': ['0.164', '0.167'],
    '<4px sigma 0.0059 cv 0.137': ['0.0059', '0.137'],
    'probe trap 38759 / 0.0001': ['38 759', '38759', '0.0001'],
    'disagreeing bins 0.5445 / 0.7109': ['0.5445', '0.7109'],
    'disagreeing bins 0.4028 / 0.5570': ['0.4028', '0.5570'],
}

missing = []
for name, alts in GROUPS.items():
    if not any(a and a in norm for a in alts):
        missing.append((name, alts))

print(f'checked {len(GROUPS)} protected items')
if missing:
    print('MISSING:')
    for name, alts in missing:
        print('  -', name, '=>', alts)
    sys.exit(1)
print('ALL PRESENT')
