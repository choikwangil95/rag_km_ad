# rag_km_ad
카카오모빌리티 광고상품 챗봇 (https://km-ad-chatbot.streamlit.app/)
## Description
### Features
- 카카오모빌리티 광고상품에 대한 설명 질문이 가능합니다.
- 광고 요구사항에 따른 광고상품 추천 요청이 가능합니다.
### Teck Stack
- LangChain - LLM App Framework
- LangSmith - LLM I/O Logging
- Streamlit - Web App Framework
### RAG Data
- https://www.kakaomobility.com/ads

## Prerequisite
- python 3.11
- poetry

## Get Started
install poetry (python package manager)
```
brew install poetry
```
run python virtual environment
```
poetry shell
```
install python package
```
poetry update
```
setting environment variables
```
.streamlit/secret.toml

OPENAI_API_KEY="<YOUR_API_KEY>"
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="<YOUR_API_KEY>"
LANGCHAIN_PROJECT="<YOUR_PROJECT_NAME>"
```
run wab app
```
streamlit run main.js
```
