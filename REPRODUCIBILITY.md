# Reproducibility Guide

## Environment
- Python 3.11
- `pip install -r requirements.txt`

## Toy stability experiment
```bash
python -m EAFCode.toy.run_toy --mode baseline --steps 300 --T 64
python -m EAFCode.toy.run_toy --mode sweep --steps 300 --T 64
python -m EAFCode.toy.plot
