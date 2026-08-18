"""One-command reproduction of every result, number and figure in the paper.

    python -m src.make_all

Order matters: step-1 TMY sizing -> validation against PVGIS -> sixteen-
year check and final designs -> results tables -> Monte Carlo -> 2-D
sweeps -> numbers.json -> paper figures -> tests. Any change to a model
parameter must be followed by this command before the paper is rebuilt.
Runtime is about ten minutes on a laptop (numba, if installed, speeds up
the multi-year loops).
"""
import subprocess, sys

STEPS = [
    ('step-1 sizing grid search (TMY, governing-year criterion)', [sys.executable, '-m', 'src.sizing']),
    ('validation against PVGIS (Upgrade 1)', [sys.executable, '-m', 'src.validation']),
    ('sixteen-year reliability and final designs (Upgrade 2)', [sys.executable, '-m', 'src.multiyear']),
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
