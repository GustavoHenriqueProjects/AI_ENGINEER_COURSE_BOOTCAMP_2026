"""ChatBot Brain4Care — interface Streamlit com suporte a texto e áudio."""

import streamlit as st

from config import MODEL, SYSTEM_PROMPT
from services import transcrever_audio, stream_chat_generator

st.set_page_config(page_title="ChatBot", page_icon=":robot_face:")
st.title("ChatBot")

# Estado da sessão
if "openai_model" not in st.session_state:
    st.session_state.openai_model = MODEL
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# Histórico do chat
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Input: texto ou áudio
prompt = st.chat_input("Escreva ou grave sua mensagem aqui", accept_audio=True)

user_message = None
if prompt:
    if isinstance(prompt, str):
        user_message = prompt.strip()
    elif getattr(prompt, "text", None):
        user_message = prompt.text.strip()
    elif getattr(prompt, "audio", None):
        with st.spinner("Transcrevendo áudio..."):
            user_message = transcrever_audio(prompt.audio.read())

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.write(user_message)

    with st.chat_message("assistant"):
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        stream = stream_chat_generator(st.session_state.openai_model, messages)
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
