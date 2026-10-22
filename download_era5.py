"""Download one year of hourly ERA5 data for a small box around Houston."""
import cdsapi
from pathlib import Path

OUT = Path("data")
OUT.mkdir(exist_ok=True)

YEAR = "2024"
AREA = [31, -97, 28, -94]          # North, West, South, East (degrees) -> ~13x13 grid at 0.25 deg

VARIABLES = {
    "t2m": "2m_temperature",        # instantaneous, Kelvin
    "tp":  "total_precipitation",   # accumulated over the hour, metres
}

client = cdsapi.Client()

for short, cds_name in VARIABLES.items():
    target = OUT / f"era5_houston_{short}_{YEAR}.nc"
    if target.exists():
        print(f"skip {target} (already downloaded)")
        continue
    print(f"requesting {cds_name} -> {target}")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": [cds_name],
            "year": [YEAR],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],   # CDS skips dates that don't exist
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(target),
    )
    print(f"done {target}")
