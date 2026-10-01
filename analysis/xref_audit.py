#!/usr/bin/env python3
"""Citation and label-resolution audit for the three builds.

Reads each .aux file and reports: \cite keys that did not resolve, \ref targets
that did not resolve, and labels defined in more than one loaded file (a duplicate
label is silently ignored by LaTeX and can point a reader at the wrong float).

Run:  python audit/xref_audit.py
"""
import re
from collections import Counter
from pathlib import Path

PR = Path(__file__).resolve().parent.parent / 'paper_PR'
BUILDS = ['paper_PR', 'paper_PR_submission', 'paper_PR_supplementary']


def main():
    for name in BUILDS:
        aux = PR / f'{name}.aux'
        log = PR / f'{name}.log'
        if not aux.exists():
            print(f'{name}: no aux')
            continue
        text = aux.read_text(encoding='utf-8', errors='replace')
        cited = re.findall(r'\\citation\{([^}]*)\}', text)
        bibcit = set(re.findall(r'\\bibcite\{([^}]*)\}', text))
        keys = [k for group in cited for k in group.split(',')]
        unresolved_cites = sorted({k for k in keys if k not in bibcit})
        refs = re.findall(r'\\newlabel\{([^}]*)\}', text)
        # \ref targets used in the log are the authoritative list of what was asked
        logtext = log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''
        asked = set(re.findall(r"Reference `([^']+)'", logtext))
        defined = set(refs)
        dupes = [k for k, v in Counter(refs).items() if v > 1]
        print(f'--- {name}')
        print(f'    cite keys used   : {len(set(keys))} distinct; undefined: {unresolved_cites or "none"}')
        print(f'    bib entries      : {len(bibcit)}')
        print(f'    labels defined   : {len(defined)}')
        print(f'    labels asked but undefined: {sorted(asked) or "none"}')
        print(f'    labels defined twice      : {dupes or "none"}')


if __name__ == '__main__':
    main()
