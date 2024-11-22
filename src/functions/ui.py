import streamlit as st
from langchain_core.messages.chat import ChatMessage


def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def clear_messages():
    st.session_state["messages"] = []
