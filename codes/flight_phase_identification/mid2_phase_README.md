# Aircraft Flight Phase Classification

## Installation

```bash
pip install google-cloud-bigquery pandas db-dtypes fire matplotlib numpy
```

## Command Line Usage

```bash
# Custom credentials path
python mid2_phase.py --credentials=/path/to/key.json

# Quick test with limited data
python mid2_phase.py --limit=10000

# Custom number of flights to visualize
python mid2_phase.py --num_flights=10
```
