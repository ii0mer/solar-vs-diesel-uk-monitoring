"""One-command reproduction of every result, number and figure in the paper.

    python -m src.make_all

Order matters: sizing sweeps -> results tables -> numbers.json -> Monte
Carlo -> 2-D sweeps -> paper figures -> tests. Any change to a model
parameter must be followed by this command before the paper is rebuilt.
"""
import subprocess, sys

STEPS = [
    ('sizing grid search (governing-year criterion)', [sys.executable, '-m', 'src.sizing']),
    ('results tables', [sys.executable, '-m', 'src.make_results']),
    ('Monte Carlo', [sys.executable, '-m', 'src.monte_carlo']),
    ('2-D sweeps', [sys.executable, '-m', 'src.sweeps_2d']),
    ('numbers.json (all quoted quantities)', [sys.executable, '-m', 'src.dump_numbers']),
    ('paper figures', [sys.executable, '-m', 'src.figures_paper']),
    ('tests', [sys.executable, '-m', 'pytest', 'tests/', '-q']),
]

if __name__ == '__main__':
    for label, cmd in STEPS:
        print(f'==> {label}')
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f'step failed: {label}')
    print('all steps complete')
