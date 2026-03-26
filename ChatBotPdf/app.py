import streamlit as st

prompt = st.chat_input("Type your message here", max_chars=100)
if prompt:
     st.write(f"User message: {prompt}")

with st.chat_message("user"):
     st.write("Hello, how are you?")

with st.chat_message("assistant"):
     st.write("I'm fine, thank you!")