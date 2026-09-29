#!/usr/bin/env python3
r"""TASK 3 checker: does every "of the main text" / "of the supplement" pointer
name a float that actually lives in the document it claims?

Ownership is read from the built .aux files, which is the only authority on
where a label resolved:

    paper_PR.aux                 the review build  (main text + supplement, inline)
    paper_PR_supplementary.aux   the standalone supplement
    paper_PR_submission.aux      the submission build (main text only)

A float is a MAIN-TEXT float if its counter value has no A/E prefix and the
label is absent from paper_PR_supplementary.aux.  It is a SUPPLEMENT float if it
carries an A-number and appears in paper_PR_supplementary.aux.

The two .aux files give a direct test: if a label's number in paper_PR.aux is
"1"/"2"/... and the label is in paper_PR_supplementary.aux with number "A..",
the float is a supplement float.  Equally, a supplement float never appears in
paper_PR_submission.aux, because \NOAPPENDIX drops it.

Usage:  python audit/check_ref_ownership.py [--verbose]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PR = ROOT / 'paper_PR'

NEWLABEL = re.compile(r'\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}')


def read_aux(name: str) -> dict:
    p = PR / name
    if not p.exists():
        raise SystemExit(f'missing {p}; build it first')
    txt = p.read_text(encoding='utf-8', errors='replace')
    # \newlabel{key}{{number}{page}{caption}{anchor}{}}
    out = {}
    for m in re.finditer(r'\\newlabel\{((?:[^{}]|\{[^{}]*\})*)\}\{\{((?:[^{}]|\{[^{}]*\})*)\}'
                         r'\{((?:[^{}]|\{[^{}]*\})*)\}', txt):
        out[m.group(1)] = m.group(2)
    return out


# Sources that are typeset into the standalone supplement PDF.
SUPP_FILES = [
    'appendices.tex', 'extra_supp.tex',
    'tables/tA1_corpus.tex', 'tables/tA2_config_audit.tex',
    'supp_tables/t1_findings.tex', 'supp_tables/t2_survey.tex',
    'supp_tables/t4_runs_per_arm.tex', 'supp_tables/t5_cross_benchmark.tex',
    'supp_tables/t7_eval_side.tex',
]
# Sources typeset into the main text only (review + submission builds).
MAIN_FILES = [
    'paper_PR.tex', 'newexp_main.tex',
    'tables/t1_findings.tex', 'tables/t2_survey.tex', 'tables/t3_noise_floor.tex',
    'tables/t4_runs_per_arm.tex', 'tables/t5_cross_benchmark.tex',
    'tables/t6_seed_vs_jitter.tex', 'tables/t7_eval_side.tex',
    'tables/t8_legality.tex', 'tables/t9_detector_ab.tex',
    'tables/t10_backward_spread.tex', 'tables/t11_amplification.tex',
    'tables/t12_dose_response.tex', 'tables/t13_equivalence.tex',
    'tables/t14_repeat_sigma.tex', 'tables/t15_matched_seed.tex',
    'tables/t16_isolated_cost.tex', 'tables/t17_e2e_cost.tex',
    'tables/t18_wallclock.tex', 'tables/t19_early_warning.tex',
    'tables/t20_full_model_dose.tex',
]

# "Table~A8", "Figure~E.2", "Table~1", "Appendix~B", "Section~5.1"
REF = re.compile(r'(?<![A-Za-z])(Table|Figure|Fig\.|Appendix|Section|Algorithm)s?'
                 r'(?:~|\s+)(A?\d+(?:\.\d+)?|[A-E])'
                 r'\s+(of the (main text|supplement|supplementary material))')


def files_containing_supp(path: Path) -> bool:
    return str(path.relative_to(PR)).replace('\\', '/') in SUPP_FILES


def main() -> int:
    verbose = '--verbose' in sys.argv
    review = read_aux('paper_PR.aux')
    supp = read_aux('paper_PR_supplementary.aux')
    submission = read_aux('paper_PR_submission.aux')

    # label -> ('supplement'|'main'), derived from the supplement's own aux
    def owner(label: str) -> str:
        if label in supp and re.match(r'^[A-E]', supp[label]):
            return 'supplement'
        return 'main'

    # number -> set(labels) in each document
    supp_by_num, main_by_num = {}, {}
    for lab, num in supp.items():
        supp_by_num.setdefault(num, set()).add(lab)
    for lab, num in review.items():
        if owner(lab) == 'main':
            main_by_num.setdefault(num, set()).add(lab)

    def kind(label: str) -> str:
        return {'tab:': 'Table', 'fig:': 'Figure', 'sec:': 'Section',
                'app:': 'Appendix'}.get(label[:4], '?')

    problems = []
    checked = 0
    for rel in SUPP_FILES + MAIN_FILES:
        p = PR / rel
        if not p.exists():
            continue
        text = p.read_text(encoding='utf-8', errors='replace')
        # ignore comment lines for reporting, but still check them separately
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            is_comment = stripped.startswith('%')
            for m in REF.finditer(line):
                word, num, phrase, doc = m.groups()
                checked += 1
                # Where does a float with this number live?
                in_supp = num in supp_by_num
                in_main = num in main_by_num
                if not in_supp and not in_main:
                    problems.append((rel, lineno, m.group(0),
                                     'number not found in any build', is_comment))
                    continue
                # A floating number is a supplement float iff it is A-prefixed.
                is_supp_float = num.startswith('A')
                claimed = 'supplement' if 'supplement' in doc else 'main'
                actual = 'supplement' if is_supp_float else 'main'
                # A figure number like E.2 is a supplement figure.
                if re.match(r'^E\.', num):
                    actual = 'supplement'
                # Lettered appendices (A..E) ship in the supplement PDF in the
                # submission package, so "of the supplement" is right for them in
                # every build and there is nothing to cross-check here.
                if word == 'Appendix':
                    if claimed != 'supplement':
                        problems.append((rel, lineno, m.group(0),
                                         'an appendix ships in the supplement and '
                                         'must be named "of the supplement"',
                                         is_comment))
                    elif verbose:
                        print(f'  ok  {rel}:{lineno}  {m.group(0)}')
                    continue
                if claimed != actual:
                    problems.append((rel, lineno, m.group(0),
                                     f'says {claimed}, actually in the {actual}',
                                     is_comment))
                elif verbose:
                    print(f'  ok  {rel}:{lineno}  {m.group(0)}')

    print(f'checked {checked} explicit "... of the main text|supplement" pointers')
    real = [x for x in problems if not x[4]]
    cmt = [x for x in problems if x[4]]
    for group, label in ((real, 'OWNERSHIP ERRORS'), (cmt, 'stale comment')):
        if group:
            print(f'\n{label}: {len(group)}')
            for rel, lineno, text, why, _ in group:
                print(f'  {rel}:{lineno}  [{why}]  {text}')
    if not real:
        print('OK: every ownership phrase matches the document the float lives in')
    return 1 if real else 0


if __name__ == '__main__':
    sys.exit(main())
