# RAG_KM_AD

카카오모빌리티 광고상품에 대한 정보를 제공하는 챗봇입니다. [서비스 링크](https://km-ad-chatbot.streamlit.app/)

## 📋 프로젝트 개요

### 문제정의
- **문제**: 150+개의 카카오모빌리티 광고상품 정보 확인이 힘들다.
- **해결**: 카카오모빌리티 광고상품 정보제공 및 추천하는 챗봇을 제공한다.

### 주요 기능

- **광고상품 설명**: 카카오모빌리티 광고상품에 대한 정보 제공 및 질문 응답
- **광고상품 추천**: 사용자의 광고 요구사항에 따른 맞춤형 광고상품 추천

### RAG 활용 데이터

- 카카오모빌리티 광고상품 데이터: [https://www.kakaomobility.com/ads](https://www.kakaomobility.com/ads)

### 인사이트
- 광고 노출 데이터를 활용하면 광고상품 추천 고도화가 가능할 것 같다.
- 챗봇 I/O 로깅 데이터를 통해 광고주 요구사항 파악이 용이할 것 같다.

## 🛠 기술 스택
### LLM
- **LangChain**: LLM 애플리케이션 프레임워크
- **LangSmith**: LLM 입력/출력 로깅
### WebApp
- **Streamlit**: 웹 애플리케이션 프레임워크

## 🚀 시작하기

### 0. 사전 요구 사항

- **Python** 3.11
- **Poetry** (Python 패키지 관리 도구)

### 1. Poetry 설치

Poetry를 설치하여 Python 패키지를 관리합니다.
```bash
brew install poetry
```

### 2. 가상 환경 실행

Poetry를 통해 Python 가상 환경을 실행합니다.

```bash
poetry shell
```

### 3. 패키지 설치

프로젝트에 필요한 Python 패키지를 설치합니다.
```bash
brew install poetry
```

### 4. 환경 변수 설정

필수 환경 변수를 설정합니다. .streamlit/secret.toml 파일을 생성하고 아래 내용을 추가하세요:
```bash
.streamlit/secret.toml

OPENAI_API_KEY="<YOUR_API_KEY>"
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="<YOUR_API_KEY>"
LANGCHAIN_PROJECT="<YOUR_PROJECT_NAME>"
```

### 5. 웹 애플리케이션 실행

Streamlit으로 웹 애플리케이션을 실행합니다.
```bash
streamlit run main.py
```
