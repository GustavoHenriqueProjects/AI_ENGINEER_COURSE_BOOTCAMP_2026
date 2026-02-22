from openai import OpenAI
import streamlit as st

st.set_page_config(page_title="ChatBot", page_icon=":robot_face:")
st.title("ChatBot")

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# Leve (~1,3 GB): ideal com 7 GB livres. Se tiver mais RAM, pode usar "llama3.2:3b" ou "llama3.2"
model = "llama3.2:1b"

if 'openai_model' not in st.session_state:
    st.session_state.openai_model = model

if 'messages' not in st.session_state:
    st.session_state.messages = []

if prompt := st.chat_input("Write your message here"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state.openai_model,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

    #& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" run llama3.2:1b