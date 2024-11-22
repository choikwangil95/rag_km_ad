import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import load_prompt
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.functions.retriever import get_retreiver
from src.functions.memory import get_session_history


def create_chain_basic():
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


@st.cache_resource(show_spinner="rag를 처리 중입니다...")
def create_chain_rag():
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt("src/assets/prompts/rag.yaml", encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name="gpt-4o", temperature=1)

    # 단계 8: 체인(Chain) 생성
    retriever = get_retreiver()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    st.session_state["chain"] = chain

    return chain


@st.cache_resource(show_spinner="rag를 처리 중입니다...")
def create_chain_with_history():

    # 프롬프트 정의
    retriever = get_retreiver()
    print(retriever)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""    
                    You are an assistant for question-answering tasks. 
                    Use the following pieces of retrieved context to answer the question.
                    Please write the main content in your answer in markdown table format.
                    If you don't know the answer, just say that you don't know. 
                    Answer in Korean.

                    #Example Format:
                    (brief summary of the answer)
                    (table)
                    (answer to the table)

                    #Context: 

                    #Answer:
                """,
            ),
            # 대화기록용 key 인 chat_history 는 가급적 변경 없이 사용하세요!
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "#Question:\n{question}"),  # 사용자 입력을 변수로 사용
        ]
    )

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name="gpt-4o", temperature=1)

    # 단계 8: 체인(Chain) 생성
    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )

    return chain_with_history
