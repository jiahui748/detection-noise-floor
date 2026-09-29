#!/usr/bin/env python3
"""Split the Pattern Recognition float set between the printed main text and
the supplementary material.

The submission must fit a hard page limit, so five display tables move into
`paper_PR/supp_tables/` and are `\\input` by `extra_supp.tex`. The moved files
are the canonical table source with exactly one change: a bare `\\ref` to a label
that stays behind in the main text becomes `\\mref`, whose number is supplied by
the generated `supp_refs.tex`. No caption claim, number or caption wording is
changed.

Run:  python audit/split_floats.py
"""
import re
import shutil
from pathlib import Path

PR = Path(__file__).resolve().parent.parent / 'paper_PR'
SRC = PR / 'tables'
DST = PR / 'supp_tables'

# The floats that move, in the order they should appear in the supplement.
MOVED = {
    't1_findings': ['tab:noise_floor', 'tab:seed_vs_jitter', 'tab:eval_side',
                    'tab:cross_benchmark', 'tab:repeat_sigma', 'tab:matched_seed',
                    'fig:grad_noise', 'fig:amplification', 'sec:operator_found'],    't2_survey': [],
    't4_runs_per_arm': ['tab:noise_floor'],
    't5_cross_benchmark': ['fig:scale_profile'],
    't7_eval_side': ['tab:seed_vs_jitter'],
}


def main():
    DST.mkdir(exist_ok=True)
    for name, external in MOVED.items():
        text = (SRC / f'{name}.tex').read_text(encoding='utf-8')
        for label in external:
            before = text
            # A main-text label referenced from the supplement goes through \mref,
            # whose number comes from the generated supp_refs.tex (see
            # audit/fix_supp_mainrefs.py). \suppRef is NOT this pass's business:
            # outside paper_PR_supplementary.tex it is undefined, so writing it
            # here would break the review build and leave the supplement printing
            # the prose "the main text" instead of a number. Only a bare \ref is
            # rewritten, and a label already routed is left alone.
            # Plain string replacement, not a regex: the label is known exactly,
            # and a regex here is one escaping mistake away from mangling the
            # prose (an earlier version wrote the word twice).  A label already
            # routed through \mref or \suppRef is left alone.
            for prefix in (r'\ref{', r'\S\ref{'):
                old = prefix + label + '}'
                new = prefix.replace(r'\ref{', r'\mref{') + label + '}'
                if old in text:
                    text = text.replace(old, new)
            if text == before:
                print(f'  note: {name}: no bare \\ref to {label}')
        # Findings carries a fixed-width p-column that overruns the text block in
        # the standalone supplement; scale it to the measure. This changes layout
        # only, never a cell.
        if name == 't1_findings' and '\\begin{adjustbox}' not in text:
            text = text.replace(r'\begin{tabular}', '\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}', 1)
            text = text.replace('\\end{tabular}\n\\end{table}',
                                '\\end{tabular}\n\\end{adjustbox}\n\\end{table}', 1)
        (DST / f'{name}.tex').write_text(text, encoding='utf-8')
        print(f'wrote supp_tables/{name}.tex')


if __name__ == '__main__':
    main()
