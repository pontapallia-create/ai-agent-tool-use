import json
import streamlit as st
from google import genai
from google.genai import types

# --- Page setup ---
st.set_page_config(page_title="AI Agent Tool Use", page_icon="🤖")
st.title("🤖 AI Agent with Tool Use (Gemini)")

# --- API key from Streamlit secrets (never hardcode it in code) ---
if "GEMINI_API_KEY" not in st.secrets:
    st.error(
        "Missing GEMINI_API_KEY. Go to Manage app -> Settings -> Secrets "
        "and add: GEMINI_API_KEY = \"your-key-here\""
    )
    st.stop()

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
MODEL_NAME = "gemini-2.0-flash"

# --- Example tool the model can call ---
def get_weather(city: str) -> str:
    # Replace this with a real weather API call if needed
    return f"The weather in {city} is sunny and 25°C."

available_functions = {"get_weather": get_weather}

# Gemini tool/function declaration
get_weather_declaration = types.FunctionDeclaration(
    name="get_weather",
    description="Get the current weather for a given city",
    parameters={
        "type": "OBJECT",
        "properties": {
            "city": {"type": "STRING", "description": "City name"}
        },
        "required": ["city"],
    },
)

tools = types.Tool(function_declarations=[get_weather_declaration])
config = types.GenerateContentConfig(tools=[tools])

# --- Chat history state (Gemini format: role "user"/"model") ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...} for display
if "history" not in st.session_state:
    st.session_state.history = []  # list of types.Content for the API

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
user_input = st.chat_input("Ask me something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.history.append(
        types.Content(role="user", parts=[types.Part(text=user_input)])
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=st.session_state.history,
                    config=config,
                )

                candidate = response.candidates[0]
                function_call = None
                for part in candidate.content.parts:
                    if part.function_call:
                        function_call = part.function_call
                        break

                if function_call:
                    # Model wants to call a tool
                    func_name = function_call.name
                    func_args = dict(function_call.args)
                    result = available_functions[func_name](**func_args)

                    # Add the model's function call turn to history
                    st.session_state.history.append(candidate.content)

                    # Add the function result back to history
                    st.session_state.history.append(
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_function_response(
                                    name=func_name,
                                    response={"result": result},
                                )
                            ],
                        )
                    )

                    # Ask the model for the final natural-language answer
                    follow_up = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=st.session_state.history,
                        config=config,
                    )
                    final_reply = follow_up.candidates[0].content.parts[0].text
                    st.markdown(final_reply)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_reply}
                    )
                    st.session_state.history.append(follow_up.candidates[0].content)
                else:
                    final_reply = candidate.content.parts[0].text
                    st.markdown(final_reply)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_reply}
                    )
                    st.session_state.history.append(candidate.content)

            except Exception as e:
                st.error(f"Error calling Gemini API: {e}")
