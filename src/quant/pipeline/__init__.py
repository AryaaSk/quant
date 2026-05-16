"""Pipeline stages: collect, structure, state, dataset, train, backtest, report.

Each stage is idempotent and skips work if its outputs are newer than its inputs.
"""
