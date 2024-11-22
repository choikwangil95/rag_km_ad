import streamlit as st

st.title("카카오모빌리티 광고상품 챗봇 💬")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for role, message in st.session_state["messages"]:
    st.chat_message(role).write(message)

prompt = st.chat_input("Say something")
if prompt:
    st.chat_message("user").write(prompt)
    st.chat_message("assistant").write(prompt)

    st.session_state["messages"].append(("user", prompt))
    st.session_state["messages"].append(("assistant", prompt))
