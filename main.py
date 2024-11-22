import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import load_prompt
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from src.functions.logging import langsmith

from dotenv import load_dotenv
import os

# API KEY 정보로드
load_dotenv()

langsmith("KM_AD_CHATBOT")

st.title("카카오모빌리티 광고상품 챗봇 💬")
openai_api_key = st.secrets["OPENAI_API_KEY"]

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# 사이드바 생성
def clear_messages():
    st.session_state["messages"] = []


with st.sidebar:
    # 초기화 버튼
    clear_btn = st.button("대화 초기화")

    if clear_btn:
        clear_messages()


def create_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 친절한 AI 어시스턴트입니다."),
            ("user", "#Question:/n{question}"),
        ]
    )

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    return chain


def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_meesage(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 이전 대화 출령
print_messages()

# 유저 인풋
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 유저 인풋 처리
if user_input:
    st.chat_message("user").write(user_input)

    chain = create_chain()
    ai_answer = chain.stream({"question": user_input})
    with st.chat_message("assistant"):
        # 컨테이너를 만들어서 스트리밍 출력
        container = st.empty()

        ai_messages = ""
        for token in ai_answer:
            ai_messages += token
            container.markdown(ai_messages)

    add_meesage("user", user_input)
    add_meesage("assistant", user_input)
