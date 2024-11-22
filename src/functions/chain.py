import streamlit as st
from src.functions.retriever import get_retreiver
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import load_prompt


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
    retriever = get_retreiver()

    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt("src/assets/prompts/rag.yaml", encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name="gpt-4o", temperature=1)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    st.session_state["chain"] = chain

    return chain
