"""MCP server exposing ERA5 Houston-area reanalysis data as tools (stdio transport)."""
import sys
from pathlib import Path

import xarray as xr
#from mcp.server.fastmcp import FastMCP
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from typing import Literal

DATA_DIR = Path(__file__).parent / "data"
MAX_ROWS = 400  # cap on timeseries rows returned, to protect the model's context window

# short name -> (unit conversion, display units, description)
VARS = {
    "t2m": (lambda da: da - 273.15, "degC", "2 m air temperature (instantaneous)"),
    "tp":  (lambda da: da * 1000.0, "mm",   "Precipitation accumulated over the hour ending at valid_time"),
}


def log(msg: str) -> None:
    print(f"[server] {msg}", file=sys.stderr)   # never stdout


def load_data() -> dict:
    data = {}
    for short, (convert, _, _) in VARS.items():
        files = sorted(DATA_DIR.glob(f"era5_houston_{short}_*.nc"))
        if not files:
            log(f"no files for {short}; tool will not offer it")
            continue
        da = xr.concat([xr.open_dataset(f)[short] for f in files], dim="valid_time")
        data[short] = convert(da).load()
        log(f"loaded {short}: {dict(da.sizes)}")
    return data


DATA = load_data()
#mcp = FastMCP("era5-houston")
mcp = MCPServer("era5-houston", version="0.1.0")

def _get_point(variable: str, lat: float, lon: float, start: str, end: str):
    if variable not in DATA:
        raise ToolError(f"Unknown variable '{variable}'. Available: {sorted(DATA)}")
    da = DATA[variable]
    lat_lo, lat_hi = float(da.latitude.min()), float(da.latitude.max())
    lon_lo, lon_hi = float(da.longitude.min()), float(da.longitude.max())
    # Refuse rather than silently snapping a far-away point to the box edge
    if not (lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi):
        raise ToolError(
            f"Point ({lat}, {lon}) is outside the data domain: "
            f"lat {lat_lo}..{lat_hi}, lon {lon_lo}..{lon_hi}"
        )
    pt = da.sel(latitude=lat, longitude=lon, method="nearest")
    pt = pt.sel(valid_time=slice(start, end))
    if pt.sizes["valid_time"] == 0:
        t0, t1 = str(da.valid_time.values[0])[:10], str(da.valid_time.values[-1])[:10]
        raise ToolError(f"No data between {start} and {end}. Coverage is {t0} to {t1}.")
    return pt


def _gridpoint(pt) -> dict:
    return {"lat": float(pt.latitude), "lon": float(pt.longitude)}


@mcp.tool()
def describe_data() -> dict:
    """List the available variables, their units, the time coverage, and the spatial domain.
    Call this first if you are unsure what data exists."""
    out = {}
    for short, da in DATA.items():
        _, units, desc = VARS[short]
        out[short] = {
            "description": desc,
            "units": units,
            "time_start": str(da.valid_time.values[0])[:16],
            "time_end": str(da.valid_time.values[-1])[:16],
            "lat_range": [float(da.latitude.min()), float(da.latitude.max())],
            "lon_range": [float(da.longitude.min()), float(da.longitude.max())],
            "grid_spacing_deg": 0.25,
            "time_step": "hourly",
            "time_zone": "UTC",
        }
    return out


@mcp.tool()
#def compute_stat(variable: str, lat: float, lon: float, start: str, end: str, stat: str) -> dict:
def compute_stat(variable: str, lat: float, lon: float, start: str, end: str,
                 stat: Literal["min", "max", "mean", "sum"]) -> dict:
    """Compute a statistic of one variable at the grid point nearest (lat, lon) over a time window.

    variable: short name from describe_data (e.g. 't2m', 'tp').
    lat, lon: decimal degrees; longitude is negative west of Greenwich (Houston is about 29.76, -95.37).
    start, end: ISO dates 'YYYY-MM-DD' (both inclusive) or datetimes 'YYYY-MM-DDTHH:MM', in UTC.
    stat: one of 'min', 'max', 'mean', 'sum'. 'sum' is only meaningful for precipitation ('tp').
    Returns the value (in the variable's units), the time it occurred for min/max,
    and the grid point actually used."""
    if stat not in ("min", "max", "mean", "sum"):
        raise ToolError("stat must be one of: min, max, mean, sum")
    if stat == "sum" and variable != "tp":
        raise ToolError(f"'sum' is not meaningful for {variable}; use mean, min, or max")
    pt = _get_point(variable, lat, lon, start, end)

    result = {"variable": variable, "units": VARS[variable][1], "stat": stat,
              "gridpoint": _gridpoint(pt), "n_hours": int(pt.sizes["valid_time"])}
    if stat in ("min", "max"):
        #i = int(pt.argmin()) if stat == "min" else int(pt.argmax())
        vals = pt.values
        i = int(vals.argmin()) if stat == "min" else int(vals.argmax())
        result["value"] = round(float(pt[i]), 2)
        result["time_of_extreme"] = str(pt.valid_time.values[i])[:16]
    else:
        result["value"] = round(float(getattr(pt, stat)()), 2)
    return result


@mcp.tool()
#def get_timeseries(variable: str, lat: float, lon: float, start: str, end: str,
#                   resolution: str = "daily") -> dict:
def get_timeseries(variable: str, lat: float, lon: float, start: str, end: str,
                   resolution: Literal["hourly", "daily"] = "daily") -> dict:
    """Return a time series of one variable at the grid point nearest (lat, lon).

    resolution: 'hourly' or 'daily'. All times are UTC; daily values use UTC calendar days
    (the daily mean for temperature, the daily total for precipitation).
    start, end: ISO dates 'YYYY-MM-DD' (both inclusive).
    At most 400 rows are returned; for long periods prefer 'daily' or use compute_stat."""
    if resolution not in ("hourly", "daily"):
        raise ToolError("resolution must be 'hourly' or 'daily'")
    pt = _get_point(variable, lat, lon, start, end)
    if resolution == "daily":
        pt = pt.resample(valid_time="1D").sum() if variable == "tp" else pt.resample(valid_time="1D").mean()

    n = int(pt.sizes["valid_time"])
    if n > MAX_ROWS:
        raise ToolError(f"{n} rows requested; limit is {MAX_ROWS}. Narrow the window or use daily resolution.")
    width = 10 if resolution == "daily" else 16
    rows = [{"time": str(t)[:width], "value": round(float(v), 2)}
            for t, v in zip(pt.valid_time.values, pt.values)]
    return {"variable": variable, "units": VARS[variable][1], "resolution": resolution,
            "gridpoint": _gridpoint(pt), "rows": rows}


if __name__ == "__main__":
    mcp.run()   # stdio transport by default
