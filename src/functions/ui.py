from src.functions.agent import get_agent_superviser
from src.functions.parser import AgentStreamParser

from src.functions.chain import create_chain_rag
import streamlit as st
from langchain_core.messages.chat import ChatMessage


def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def clear_messages():
    st.session_state["messages"] = []


def print_sidebar():
    with st.sidebar:
        st.header("카카오모빌리티 광고상품 챗봇", divider="gray")
        st.markdown(
            """
            카카오 모빌리티와 광고 상품에 대한 정보를 제공하는 챗봇입니다.
            """
        )

        st.subheader("질문 예시", divider="gray")
        st.markdown(
            """
            - 광고상품 설명 및 추천
                - 카카오모빌리티 디스플레이 광고상품에 대해 설명해줘
                - 예산 5,000만원으로 집행 가능한 택시 광고상품을 5개 추천해줘
                - 300만 이상 노출 (1주일 기준) 이 가능한 광고상품을 추천해줘
            - 웹 검색
                - 카카오모빌리티 과징금 얼마야?
                - 카카오모빌리티 채용에 대해 알려줘
            - 리포트 분석
                - T.B.D
            """
        )

        st.subheader("설정", divider="gray")
        # 초기화 버튼
        clear_btn = st.button("대화 초기화")
        if clear_btn:
            clear_messages()

        st.markdown(
            """
            <style>
            .sticky-footer {
                position: fixed;
                bottom: 0;
                color: white
                text-align: center;
                padding: 10px;
                font-size: 16px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # 고정된 텍스트를 하단에 표시
        st.markdown(
            '<a href="https://github.com/choikwangil95/rag_km_ad" class="sticky-footer"><img src="https://img.shields.io/badge/GitHub-000000?style=flat&logo=github"></a>',
            unsafe_allow_html=True,
        )


def print_user_input():
    chain = create_chain_rag()

    user_input = st.chat_input("궁금한 내용을 물어보세요!")

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


def print_user_input_with_agent():
    agent_superviser = get_agent_superviser()
    agent_stream_parser = AgentStreamParser()
    user_input = st.chat_input("궁금한 내용을 물어보세요!")

    if user_input:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = agent_superviser.stream(
            {"input": user_input},
            # session_id 설정
            config={"configurable": {"session_id": st.session_state['user_id']}},
        )
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for step in response:
                # print(step, agent_stream_parser.process_agent_steps(step))
                if agent_stream_parser.process_agent_steps(step) is not None:
                    ai_answer += agent_stream_parser.process_agent_steps(step)
                    print('==='*20, ai_answer, '==='*20)
                    container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
