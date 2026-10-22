"""Hand-written agent loop: Gemini on Vertex AI + the ERA5 MCP server over stdio."""
import argparse
import asyncio
import json
import sys
import time

from google import genai
from google.genai import types
from google.genai import errors
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT = "wx-agent-0"
LOCATION = "global"
MAX_STEPS = 8   # hard cap on model calls per question: runaway-loop and cost guard

SYSTEM = """You answer questions about weather using ONLY the provided tools, which query
ERA5 reanalysis data for a small region around Houston, Texas.
- Never guess or use outside knowledge for numbers; every number must come from a tool result.
- Tool timestamps are UTC. When you report a time, give it in UTC and in Houston local time
  (2024: CST = UTC-6 before 2024-03-10 and after 2024-11-03; CDT = UTC-5 in between).
- Always state units.
- If a request is outside the data's coverage, say so plainly instead of approximating.
- Keep the final answer short."""


def mcp_tools_to_gemini(mcp_tools) -> types.Tool:
    """Translate MCP tool definitions into Gemini function declarations (schemas pass through as-is)."""
    return types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name=t.name,
            description=t.description or "",
            parameters_json_schema=t.input_schema,
        )
        for t in mcp_tools
    ])


async def run_agent(question: str, model: str, session: ClientSession,
                    gemini_tool: types.Tool, client: genai.Client, verbose: bool = True) -> dict:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        tools=[gemini_tool],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.0,
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    trace = []
    tokens = {"prompt": 0, "output": 0, "thinking": 0}
    t0 = time.perf_counter()

    for step in range(1, MAX_STEPS + 1):
        resp = await client.aio.models.generate_content(model=model, contents=contents, config=config)

        u = resp.usage_metadata
        tokens["prompt"] += u.prompt_token_count or 0
        tokens["output"] += u.candidates_token_count or 0
        tokens["thinking"] += u.thoughts_token_count or 0

        # Keep the model's full turn in the history. This also carries any "thought signatures",
        # which newer Gemini models require to be sent back unchanged on the next call.
        contents.append(resp.candidates[0].content)

        calls = resp.function_calls or []
        if not calls:   # no tool requests -> this is the final answer
            return {"question": question, "model": model, "answer": resp.text,
                    "model_calls": step, "tool_calls": trace, "tokens": tokens,
                    "latency_s": round(time.perf_counter() - t0, 2), "stopped": "answer"}

        response_parts = []
        for fc in calls:
            args = dict(fc.args or {})
            result = await session.call_tool(fc.name, args)
            text = result.content[0].text if result.content else ""
            trace.append({"step": step, "tool": fc.name, "args": args,
                          "is_error": result.is_error, "result": text})
            if verbose:
                flag = "ERROR" if result.is_error else "ok"
                print(f"[step {step}] {fc.name}({json.dumps(args)}) -> {flag}", file=sys.stderr)
            payload = {"error": text} if result.is_error else {"result": text}
            response_parts.append(types.Part.from_function_response(name=fc.name, response=payload))

        contents.append(types.Content(role="user", parts=response_parts))

    return {"question": question, "model": model, "answer": None,
            "model_calls": MAX_STEPS, "tool_calls": trace, "tokens": tokens,
            "latency_s": round(time.perf_counter() - t0, 2), "stopped": "max_steps"}


async def main():
    ap = argparse.ArgumentParser(description="Ask the ERA5 weather agent a question.")
    ap.add_argument("question")
#    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--location", default=LOCATION)
    args = ap.parse_args()

    #client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    client = genai.Client(vertexai=True, project=PROJECT, location=args.location)
    params = StdioServerParameters(command=sys.executable, args=["server.py"])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools

            try:
                result = await run_agent(args.question, args.model, session,
                                         mcp_tools_to_gemini(mcp_tools), client)
            except errors.APIError as e:
                print(f"\nModel call failed: {e}", file=sys.stderr)
                return

    print("\n=== ANSWER ===")
    print(result["answer"] if result["answer"] else f"(no answer: stopped at {result['stopped']})")
    print("\n=== RUN SUMMARY ===")
    print(json.dumps({k: result[k] for k in ("model", "model_calls", "tokens", "latency_s", "stopped")}, indent=2))
    print(f"tool calls: {len(result['tool_calls'])}")


if __name__ == "__main__":
    asyncio.run(main())
