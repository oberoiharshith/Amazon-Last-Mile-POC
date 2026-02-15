# Last Mile Simulation + ML Dispatch POC

This repo is a small but end-to-end POC for *last mile* network analytics:
- a discrete event simulator for deliveries (orders, stations, drivers, traffic)
- an ML model that predicts delay risk from operational features
- a dispatch policy that uses delay risk to make better assignments
- an experiment runner that compares **baseline** vs **ML-aware** dispatch on the same random seeds
- KPI + stats output (SLA %, avg delivery time, cost per package, utilization)

It’s intentionally compact so a hiring manager can skim it fast, run it locally, and see results.

## What problem this models
Last mile operations constantly trade off:
- customer promise (on-time delivery / SLA)
- cost (miles, overtime, under/over-utilization)
- capacity constraints (driver shifts, station limits)
- uncertainty (demand spikes, traffic variability)

This POC shows how you can:
1) simulate the system,
2) learn delay risk,
3) change decisions using the learned signal,
4) measure impact with a controlled experiment.

## Quickstart

### 1) Create env + install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run the experiment
```bash
python run_experiment.py --days 14 --drivers 60 --stations 6 --seed 7
```

Outputs:
- `reports/results_summary.json`
- `reports/results_runs.csv`
- `reports/plots/*.png`
- `reports/executive_summary.pdf`

## Repo layout
- `src/simulation_engine.py` discrete event simulator
- `src/delay_model.py` model training + evaluation
- `src/dispatch_optimizer.py` baseline + ML-aware dispatch policies
- `src/experiment_framework.py` A/B runner + significance tests
- `src/metrics.py` KPIs
- `run_experiment.py` CLI entrypoint

## Notes
- Data is synthetic (generated from plausible distributions).
- This is not a routing solver. It’s a dispatch/assignment problem with realistic constraints.
- The point is the *workflow*: simulation → model → policy → experiment → KPIs.

## Why it’s relevant
It demonstrates:
- translating an ambiguous ops problem into a measurable model
- discrete event simulation fundamentals
- experimentation and impact measurement
- production-style modular code (not a one-off notebook)

