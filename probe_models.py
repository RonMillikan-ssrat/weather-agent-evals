"""Which Gemini models can this project call, where, and do they do tool calling?"""
from google import genai
from google.genai import types

PROJECT = "wx-agent-0"
LOCATIONS = ["us-central1", "global"]
MODELS = ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.6-flash",
          "gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

weather_tool = types.Tool(function_declarations=[types.FunctionDeclaration(
    name="get_temperature",
    description="Get the current temperature for a city.",
    parameters_json_schema={"type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"]},
)])
tool_config = types.GenerateContentConfig(
    tools=[weather_tool],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

print(f"{'model':<24}{'location':<13}{'plain call':<12}{'tool call'}")
for loc in LOCATIONS:
    client = genai.Client(vertexai=True, project=PROJECT, location=loc)
    for m in MODELS:
        try:
            client.models.generate_content(model=m, contents="Reply: ok")
            plain = "ok"
        except Exception as e:
            print(f"{m:<24}{loc:<13}{'FAIL':<12}{str(e)[:70]}")
            continue
        try:
            r = client.models.generate_content(
                model=m, contents="What's the temperature in Houston? Use the tool.", config=tool_config)
            calls = r.function_calls or []
            tool = f"ok -> {calls[0].name}({dict(calls[0].args)})" if calls else "no call made"
        except Exception as e:
            tool = f"FAIL {str(e)[:60]}"
        print(f"{m:<24}{loc:<13}{plain:<12}{tool}")
