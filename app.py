import json
import streamlit as st
from openai import OpenAI

# --- Page setup ---
st.set_page_config(page_title="AI Agent Tool Use", page_icon="🤖")
st.title("🤖 AI Agent with Tool Use")

# --- API key from Streamlit secrets (never hardcode it in code) ---
if "OPENAI_API_KEY" not in st.secrets:
    st.error(
        "Missing OPENAI_API_KEY. Go to Manage app -> Settings -> Secrets "
        "and add: OPENAI_API_KEY = \"sk-...\""
    )
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Example tool the model can call ---
def get_weather(city: str) -> str:
    # Replace this with a real weather API call if needed
    return f"The weather in {city} is sunny and 25°C."

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        },
    }
]

available_functions = {"get_weather": get_weather}

# --- Chat history state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render past messages
for msg in st.session_state.messages:
    if msg["role"] in ("user", "assistant") and msg.get("content"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# --- Chat input ---
user_input = st.chat_input("Ask me something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.messages,
                    tools=tools,
                )
                msg = response.choices[0].message

                # If the model wants to call a tool
                if msg.tool_calls:
                    st.session_state.messages.append(msg.model_dump())

                    for tool_call in msg.tool_calls:
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)
                        result = available_functions[func_name](**func_args)

                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        })

                    # Get the final natural-language response after tool execution
                    follow_up = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=st.session_state.messages,
                    )
                    final_reply = follow_up.choices[0].message.content
                    st.markdown(final_reply)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_reply}
                    )
                else:
                    st.markdown(msg.content)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": msg.content}
                    )
            except Exception as e:
                st.error(f"Error calling OpenAI API: {e}")
