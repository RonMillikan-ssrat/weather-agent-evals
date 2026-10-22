"""Exercise the MCP server the same way the agent will: as a subprocess over stdio."""
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HOUSTON = {"lat": 29.76, "lon": -95.37}

CALLS = [
    ("describe_data", {}),
    ("compute_stat", {"variable": "t2m", **HOUSTON, "start": "2024-07-01", "end": "2024-07-31", "stat": "max"}),
    ("compute_stat", {"variable": "t2m", **HOUSTON, "start": "2024-01-01", "end": "2024-12-31", "stat": "min"}),
    ("get_timeseries", {"variable": "t2m", **HOUSTON, "start": "2024-01-13", "end": "2024-01-19"}),
    # Deliberate failures: the model must be able to read and recover from these
    ("compute_stat", {"variable": "t2m", "lat": 40.0, "lon": -105.0, "start": "2024-07-01", "end": "2024-07-31", "stat": "max"}),
    ("compute_stat", {"variable": "t2m", **HOUSTON, "start": "2024-07-01", "end": "2024-07-31", "stat": "sum"}),
]


async def main():
    params = StdioServerParameters(command=sys.executable, args=["server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("=== TOOLS THE MODEL WILL SEE ===")
            for t in tools.tools:
                print(f"\n{t.name}: {(t.description or '').splitlines()[0]}")
                print("  schema:", json.dumps(t.input_schema))

            print("\n=== CALLS ===")
            for name, args in CALLS:
                r = await session.call_tool(name, args)
                print(f"\n{name} {args}\n  is_error={r.is_error}\n  {r.content[0].text}")


asyncio.run(main())
