# 🤖 AI Agent — Tool Use

A conversational AI agent built with Google Gemini that can decide,
on its own, when to call external tools instead of only answering
from what it already knows — a core building block of agentic AI
systems.

---

## 🚀 Live Demo

👉 https://ai-agent-tool-use-cxyqyzpactangjdzs6igqc.streamlit.app/

Try the agent directly in your browser — ask it to calculate something,
check the weather, or tell you the date.

---

## 📌 Project Overview

Unlike a plain chatbot, this agent has access to a set of real
Python functions ("tools"). When you ask a question, Gemini decides:

- Can I answer this directly? → it just answers.
- Do I need a tool to get an accurate/live answer? → it calls the
  right tool automatically, reads the result, and replies using it.

This is called **function calling** or **tool use**, and it's the
same underlying pattern used to build real AI agents that can browse
the web, query databases, control apps, and more.

---

## 🛠️ Available Tools

| Tool | What it does |
|---|---|
| `calculator` | Evaluates arithmetic expressions |
| `get_current_datetime` | Returns the current date and time |
| `count_words` | Counts words/characters in a piece of text |
| `get_weather` | Fetches live weather for any city (no API key needed) |
| `convert_temperature` | Converts between Celsius, Fahrenheit, Kelvin |

---

## 🧠 How It Works

```text
                 USER MESSAGE
                       │
                       ▼
              Gemini Chat Session
                       │
        ┌──────────────┴──────────────┐
        │                              │
        ▼                              ▼
  Answers directly            Decides a tool is needed
  (no tool needed)                     │
                                        ▼
                             Calls the matching Python
                             function automatically
                                        │
                                        ▼
                             Reads the tool's result
                                        │
                                        ▼
                             Generates final answer
                             using that result
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Add your Gemini API key either as an environment variable:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

or in `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your_api_key_here"
```

On **Streamlit Community Cloud**, add `GOOGLE_API_KEY` under
**App Settings → Secrets** instead.

---

## 💬 Example Questions to Try

- "What's 45 * 12 + 8?"
- "What's the weather in Hyderabad right now?"
- "What day is it today?"
- "Convert 100°F to Celsius"
- "How many words are in this sentence: 'The quick brown fox jumps'?"
- "What's the weather in Tokyo, and what's that in Celsius if it's shown in Fahrenheit?" *(tests multi-tool use in one question)*

---

## 📁 Project Structure

```
ai-agent-tool-use/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Contact

For questions or feedback, reach out at: ponthapalliarun@gmail.com
