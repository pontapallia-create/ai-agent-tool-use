"""
AI Agent with Tool Use — Streamlit Web App
---------------------------------------------------
A conversational AI agent that can decide, on its own, when to call
external tools (calculator, weather, date/time, word counter,
temperature converter) to answer a question — instead of only
relying on what it already knows.

Author: Ponthapalli Arun Kumar
"""

import ast
import operator
from datetime import datetime

import requests
import streamlit as st

from google import genai
from google.genai import types


# =========================================================
# Configuration
# =========================================================

CHAT_MODEL = "gemini-3.6-flash"


# =========================================================
# Streamlit Page Configuration
# =========================================================

st.set_page_config(
    page_title="AI Agent — Tool Use",
    page_icon="🤖",
    layout="centered"
)


# =========================================================
# Gemini Client
# =========================================================

def get_gemini_client() -> genai.Client:
    """
    Create Gemini client using a secret/API key.

    Local:
        .streamlit/secrets.toml

    Streamlit Cloud:
        App Settings -> Secrets
    """

    api_key = (
        st.secrets.get("GOOGLE_API_KEY")
        or __import__("os").environ.get("GOOGLE_API_KEY")
    )

    if not api_key:
        st.error(
            "No Google API key found. "
            "Add GOOGLE_API_KEY to Streamlit Secrets "
            "or set it as an environment variable."
        )
        st.stop()

    return genai.Client(api_key=api_key)


# =========================================================
# TOOLS
# ---------------------------------------------------------
# Each tool is a plain Python function with a docstring and
# type hints. The Gemini SDK reads these automatically to
# decide WHEN and HOW to call each one — no manual schema
# needed.
# =========================================================

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](
            _safe_eval(node.left),
            _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported or unsafe expression.")


def calculator(expression: str) -> str:
    """
    Evaluates a basic arithmetic expression and returns the result.

    Supports +, -, *, /, %, ** and parentheses. Use this whenever
    the user asks a math question or needs a calculation done.

    Args:
        expression: A math expression as a string, e.g. "12 * (3 + 4)"
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"The result of {expression} is {result}."
    except Exception:
        return f"Could not evaluate the expression: {expression}"


def get_current_datetime() -> str:
    """
    Returns the current date and time.

    Use this whenever the user asks what the date, day, or time is.
    """
    now = datetime.now()
    return now.strftime("Today is %A, %B %d, %Y and the time is %I:%M %p.")


def count_words(text: str) -> str:
    """
    Counts the number of words and characters in a piece of text.

    Use this when the user asks how long a piece of text is, or asks
    for a word/character count.

    Args:
        text: The text to analyze.
    """
    words = len(text.split())
    chars = len(text)
    return f"That text has {words} word(s) and {chars} character(s)."


def get_weather(city: str) -> str:
    """
    Gets the current real-world weather for a given city.

    Use this whenever the user asks about the weather, temperature,
    or conditions in a specific place.

    Args:
        city: The name of the city, e.g. "Hyderabad" or "London"
    """
    try:
        response = requests.get(
            f"https://wttr.in/{city}?format=%C+%t+(feels+like+%f)",
            timeout=6
        )
        if response.status_code == 200 and response.text.strip():
            return f"Current weather in {city}: {response.text.strip()}"
        return f"Could not retrieve weather for {city}."
    except requests.RequestException:
        return f"Weather service is unavailable right now for {city}."


def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """
    Converts a temperature value between Celsius, Fahrenheit, and Kelvin.

    Use this when the user asks to convert a temperature, e.g.
    "convert 100 F to C".

    Args:
        value: The numeric temperature value.
        from_unit: The unit to convert from ("C", "F", or "K").
        to_unit: The unit to convert to ("C", "F", or "K").
    """
    from_unit = from_unit.strip().upper()[:1]
    to_unit = to_unit.strip().upper()[:1]

    # Normalize to Celsius first
    if from_unit == "C":
        celsius = value
    elif from_unit == "F":
        celsius = (value - 32) * 5 / 9
    elif from_unit == "K":
        celsius = value - 273.15
    else:
        return f"Unknown unit: {from_unit}"

    if to_unit == "C":
        result = celsius
    elif to_unit == "F":
        result = celsius * 9 / 5 + 32
    elif to_unit == "K":
        result = celsius + 273.15
    else:
        return f"Unknown unit: {to_unit}"

    return f"{value}°{from_unit} is {round(result, 2)}°{to_unit}."


TOOLS = [
    calculator,
    get_current_datetime,
    count_words,
    get_weather,
    convert_temperature,
]


# =========================================================
# Streamlit UI
# =========================================================

def main():

    client = get_gemini_client()

    st.title("🤖 AI Agent — Tool Use")

    st.caption(
        "An AI agent that decides on its own when to call real tools — "
        "calculator, live weather, date/time, and more — instead of "
        "just guessing an answer."
    )

    with st.sidebar:
        st.header("🛠️ Available Tools")
        for tool in TOOLS:
            st.markdown(f"**{tool.__name__}**")
            first_line = (tool.__doc__ or "").strip().split("\n")[0]
            st.caption(first_line)
            st.divider()

        st.header("⚙️ Configuration")
        st.write(f"**Chat model:** `{CHAT_MODEL}`")

        if st.button("🔄 Reset conversation"):
            st.session_state.pop("chat_history", None)
            st.session_state.pop("chat_session", None)
            st.rerun()

    # -----------------------------------------------------
    # Persistent chat session (keeps tool-use context)
    # -----------------------------------------------------
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = client.chats.create(
            model=CHAT_MODEL,
            config=types.GenerateContentConfig(
                tools=TOOLS,
                system_instruction=(
                    "You are a helpful AI agent with access to tools "
                    "(calculator, weather, date/time, word counter, "
                    "temperature converter). Use a tool whenever it "
                    "would give a more accurate or up-to-date answer "
                    "than your own knowledge. Otherwise, answer directly."
                ),
            ),
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # -----------------------------------------------------
    # Render past messages
    # -----------------------------------------------------
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # -----------------------------------------------------
    # Chat input
    # -----------------------------------------------------
    example = "Try: 'What's 45 * 12?' or 'Weather in Hyderabad?'"
    user_input = st.chat_input(example)

    if user_input:

        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking (and calling tools if needed)..."):
                try:
                    response = st.session_state.chat_session.send_message(
                        user_input
                    )
                    answer = response.text or (
                        "I couldn't generate a response. Please try again."
                    )
                except Exception as error:
                    answer = f"Error: {error}"

            st.markdown(answer)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )


if __name__ == "__main__":
    main()