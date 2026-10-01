import re
from pathlib import Path

PR = Path('paper_PR')
for build in ('paper_PR_submission', 'paper_PR', 'paper_PR_supplementary'):
    aux = (PR / f'{build}.aux').read_text(encoding='utf-8', errors='replace')
    bbl = (PR / f'{build}.bbl').read_text(encoding='utf-8', errors='replace')
    cites = set()
    for c in re.findall(r'\\citation\{([^}]*)\}', aux):
        cites.update(x.strip() for x in c.split(',') if x.strip())
    bib = set(re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}', bbl))
    labels = set(re.findall(r'\\newlabel\{([^}]*)\}', aux))
    refs = set()
    for r in re.findall(r'\\@setckpt|', aux):
        pass
    log = (PR / f'{build}.log').read_text(encoding='utf-8', errors='replace')
    undef_ref = sorted(set(re.findall(r"Reference `([^']+)' on page", log)))
    undef_cit = sorted(set(re.findall(r"Citation `([^']+)' on page", log)))
    print(f'--- {build} ---')
    print(f'  cite keys: {len(cites)}   bibitems: {len(bib)}')
    print(f'  cited-not-in-bbl: {sorted(cites - bib)}')
    print(f'  in-bbl-not-cited: {sorted(bib - cites)}')
    print(f'  undefined references in log: {undef_ref}')
    print(f'  undefined citations in log: {undef_cit}')
    print(f'  labels defined: {len(labels)}')
