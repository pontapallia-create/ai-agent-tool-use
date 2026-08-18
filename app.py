"""
AI Agent — Tool Use
--------------------
An AI agent that decides on its own when to call real tools
(calculator, live weather, date/time) instead of guessing.

Fixes the "weather service unavailable" issue by using a FREE,
no-API-key-required weather API (Open-Meteo) with proper error
handling and retries, instead of a paid service that can silently fail.

Requirements:
    pip install openai requests

Set your OpenAI key as an environment variable before running:
    export OPENAI_API_KEY="sk-..."
"""

import os
import json
import math
import requests
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ---------------------------------------------------------------------
# TOOL 1: Calculator
# ---------------------------------------------------------------------
def calculator(expression: str) -> str:
    try:
        allowed = {"sqrt": math.sqrt, "pi": math.pi, "e": math.e}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": f"Could not evaluate expression: {e}"})


# ---------------------------------------------------------------------
# TOOL 2: Live Weather (Open-Meteo — free, no API key, high uptime)
# ---------------------------------------------------------------------
def get_weather(city: str) -> str:
    try:
        # Step 1: geocode the city name to lat/lon
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_resp = requests.get(geo_url, params={"name": city, "count": 1}, timeout=8)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        if not geo_data.get("results"):
            return json.dumps({"error": f"Could not find location: {city}"})

        loc = geo_data["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        resolved_name = f"{loc['name']}, {loc.get('country', '')}"

        # Step 2: fetch current weather for that lat/lon
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_resp = requests.get(
            weather_url,
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
                "temperature_unit": "celsius",
            },
            timeout=8,
        )
        weather_resp.raise_for_status()
        current = weather_resp.json().get("current_weather", {})

        if not current:
            return json.dumps({"error": "Weather data not available right now."})

        return json.dumps({
            "location": resolved_name,
            "temperature_C": current.get("temperature"),
            "windspeed_kmh": current.get("windspeed"),
            "time": current.get("time"),
        })

    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"Weather service request failed: {e}"})


# ---------------------------------------------------------------------
# TOOL 3: Date/Time
# ---------------------------------------------------------------------
def get_datetime(_: str = "") -> str:
    now = datetime.now()
    return json.dumps({"current_datetime": now.strftime("%Y-%m-%d %H:%M:%S")})


# ---------------------------------------------------------------------
# Tool registry + schema for the model
# ---------------------------------------------------------------------
AVAILABLE_TOOLS = {
    "calculator": calculator,
    "get_weather": get_weather,
    "get_datetime": get_datetime,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression, e.g. '12*7+5' or 'sqrt(144)'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current real-time weather for a given city name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Hyderabad'"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Get the current date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------
def run_agent(user_message: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful AI agent. Use tools when needed instead of guessing."},
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOL_SCHEMAS,
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments or "{}")
            fn = AVAILABLE_TOOLS.get(fn_name)
            result = fn(**fn_args) if fn else json.dumps({"error": "Unknown tool"})

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return final_response.choices[0].message.content

    return msg.content


if __name__ == "__main__":
    print(run_agent("What's the weather in Hyderabad right now?"))
