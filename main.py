from src.functions.chain import create_chain_rag
from src.functions.ui import add_message, clear_messages, print_messages
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

# 로깅
langsmith("KM_AD_CHATBOT")

# 세션 스테이트 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("카카오모빌리티 광고상품 챗봇 💬")
openai_api_key = st.secrets["OPENAI_API_KEY"]


# 사이드바 생성
with st.sidebar:
    # 초기화 버튼
    clear_btn = st.button("대화 초기화")

    if clear_btn:
        clear_messages()


# 이전 대화 출력
print_messages()

# 유저 인풋
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

chain = create_chain_rag()

# 유저 인풋 처리
if user_input:
    # 사용자의 입력
    st.chat_message("user").write(user_input)
    # 스트리밍 호출
    response = chain.stream(user_input)
    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대화기록을 저장한다.
    add_message("user", user_input)
    add_message("assistant", ai_answer)
