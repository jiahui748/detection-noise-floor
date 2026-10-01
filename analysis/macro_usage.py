#!/usr/bin/env python3
"""Count how many times each \\survey... macro is expanded across the sources, so
a compression pass can tell whether removing a sentence orphans a measured value.
A macro defined in the generated block but used nowhere would take its number out
of the printed paper.
"""
import re
from pathlib import Path

PR = Path(__file__).resolve().parent.parent / 'paper_PR'
FILES = ['paper_PR.tex', 'appendices.tex', 'extra_supp.tex']
FILES += [str(p.relative_to(PR)) for p in sorted((PR / 'tables').glob('*.tex'))]
FILES += [str(p.relative_to(PR)) for p in sorted((PR / 'supp_tables').glob('*.tex'))]

text = {}
for f in FILES:
    p = PR / f
    if p.exists():
        text[f] = p.read_text(encoding='utf-8', errors='replace')
allsrc = '\n'.join(text.values())

main = (PR / 'paper_PR.tex').read_text(encoding='utf-8', errors='replace')
defined = sorted(set(re.findall(r'\\newcommand\{\\(survey\w+)\}', main)))
print(f'{len(defined)} \\survey macros defined in paper_PR.tex\n')
orphans = []
for name in defined:
    uses = len(re.findall(r'\\' + name + r'(?![a-zA-Z])', allsrc))
    # minus the definition itself
    print(f'  {uses - 1:>3} use(s)  \\{name}')
    if uses - 1 <= 0:
        orphans.append(name)
print()
print('ORPHANED (value would disappear):', orphans or 'none')
