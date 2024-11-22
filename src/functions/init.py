import streamlit as st
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
import uuid
from src.functions.chain import create_chain_rag, create_chain_with_history
from src.functions.logging import langsmith


def init():
    # 인메모리 캐시를 사용합니다.
    set_llm_cache(InMemoryCache())

    # API KEY 정보로드
    load_dotenv()

    # 로깅
    langsmith("KM_AD_CHATBOT")

    # 세션 스테이트 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "store" not in st.session_state:
        st.session_state["store"] = {}

    if "chain" not in st.session_state:
        st.session_state["chain"] = None

    # 접속 유저마다 고유한 ID 생성 및 세션 상태에 저장
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
