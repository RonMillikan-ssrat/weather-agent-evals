# weather-agent-evals

A tool-using LLM agent that queries ERA5 reanalysis data through an MCP server,
scored against a deterministic eval suite with ground-truth answers computed
directly from the data.

## Status

| Component | State |
|---|---|
| MCP server (`server.py`): `describe_data`, `compute_stat`, `get_timeseries` | Done |
| Agent loop (hand-written, no framework) | Done |
| Eval suite (ground truth from xarray) | Done |
| Local vs. API model comparison | Done |

## Data

Hourly ERA5 single-level data (2 m temperature, total precipitation) for 2024
over a 3°×3° box around Houston (28–31°N, 94–97°W), 0.25° grid.
All times are UTC.

## Setup

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python download_era5.py     # requires a CDS account and ~/.cdsapirc
python test_server.py       # exercises the MCP server over stdio
```
