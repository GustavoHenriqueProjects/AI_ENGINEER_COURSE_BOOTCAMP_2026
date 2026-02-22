import streamlit as st

st.title("ChatBot")

st.title("_This_ is :blue[a title] :speech_balloon:")

st.header("This is a header")

st.subheader("This is a subheader")

st.text("This is a text")

st.write("This is a text")

st.markdown("# This is a markdown \n **This is a bold text** \n- This is a list item")

data = {"name": "John", "age": 30, "city": "New York"}
st.write(data)


with st.chat_message("human"):
     st.write("Hello, how are you?")

     prompt = st.chat_input("Enter a message", max_chars=100)
     if prompt:
        st.write(f"User message: {prompt}")


st.title("Buttons")

if 'show_second_button' not in st.session_state:
    st.session_state.show_second_button = False

if 'second_clicked' not in st.session_state:
    st.session_state.second_clicked = False

if st.button("First Button"):
    st.write("Button clicked")
    st.session_state.show_second_button = True

if st.session_state.show_second_button:
    if st.button("Second Button", type="primary"):
        st.session_state.second_clicked = True
        st.write("Second Button clicked")        

if st.session_state.second_clicked:
    st.write("🎉 O segundo botão foi ativado e o estado foi salvo!")
