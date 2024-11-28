# RAG_KM_AD

카카오모빌리티 광고상품에 대한 정보를 제공하는 챗봇입니다. [서비스 링크](https://km-ad-chatbot.streamlit.app/)

![rag-graphic-2 (1)](https://github.com/user-attachments/assets/eab523bf-a423-4358-a65e-8789eb3c3ec0)

## 📑 목차
- [1 📋 프로젝트 개요](#-프로젝트-개요)
- [2 🛠 기술 스택](#-기술-스택)
- [3 🔍 RAG 프로세스](#-rag-프로세스)
- [4 🔍 RAG 모니터링](#-rag-모니터링)
- [5 🚀 시작하기](#-시작하기)

## 📋 프로젝트 개요

### 문제정의
- **문제**: 150+개의 카카오모빌리티 광고상품 정보 확인이 힘들다.
- **해결**: 카카오모빌리티 광고상품 정보제공 및 추천하는 챗봇을 제공한다.

### 주요 기능

- **광고상품 설명**: 카카오모빌리티 광고상품에 대한 정보 제공 및 질문 응답
- **광고상품 추천**: 사용자의 광고 요구사항에 따른 맞춤형 광고상품 추천

### 인사이트
- 광고 노출 데이터를 활용하면 광고상품 추천 고도화가 가능할 것 같다.
- 챗봇 I/O 로깅 데이터를 통해 광고주 요구사항 파악이 용이할 것 같다.

## 🛠 기술 스택
### LLM
- **LangChain**: LLM 애플리케이션 프레임워크
- **LangSmith**: LLM 입력/출력 로깅
### WebApp
- **Streamlit**: 웹 애플리케이션 프레임워크

### 활용 데이터
- 카카오모빌리티 광고상품 데이터: [https://www.kakaomobility.com/ads](https://www.kakaomobility.com/ads)

## 🔍 RAG 프로세스
Retrieval-Augmented Generation(RAG)는 기존의 언어 모델의 한계를 넘어서 정보 검색과 생성을 통합하는 방법론입니다.

아래의 전처리 각 단계별로 적절히 엔지니어링하여 RAG의 성능 및 효율을 향상시킬 수 있습니다.

![image](https://github.com/user-attachments/assets/fc43049e-1320-4c31-8792-b538def8cc4d)

### 1. 도큐먼트 로드 (Document Loader)

외부 데이터 소스에서 필요한 문서를 로드하고 초기 처리를 합니다.

```python
# 단계 1: 문서 로드

# 텍스트 로더 생성
file_path = "src/assets/txt/AD_INFO.txt"
loader = TextLoader(file_path)

# 문서 로드
document = loader.load()
```

### 2. 텍스트 분할 (Text Splitter)

LLM 모델에는 각각 입/출력 토큰수 제한이 존재함

따라서, 로드된 문서를 처리 가능한 작은 단위로 분할합니다. 큰 책을 챕터별로 나누는 것과 유사합니다.

```python
# 단계 2: 문서 분할

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=100
)

split_documents = text_splitter.split_documents([document])
```

- chunk-size: 문서 분할 문자 수
- chunk-overlap: 분할된 문서 교집합 영역

### 3. 임베딩 (Embedding), 벡터스토어(Vector Store) 저장

각 문서 또는 문서의 일부를 벡터 형태로 변환하여, 기계가 이해할 수 있는 수치적 형태로 변환

임베딩된 벡터들을 데이터베이스에 저장합니다. 이는 요약된 키워드를 색인화하여 나중에 빠르게 찾을 수 있도록 하는 과정입니다.

```python
# 단계 3: 임베딩(Embedding) 객체 생성

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```
```python
# 단계 4: DB 생성(Create DB) 및 저장

vectorstore = FAISS.from_documents(
    documents=all_split_documents, embedding=embeddings
)

# document_chunked_1 = [0.02678847312927246, 0.03230374678969383, -0.02981565333902836, ...]
# document_chunked_2 = [-0.020303381606936455, 0.015657879412174225, -0.059232909232378006, ...]
# document_chunked_3 = [-0.005731410812586546, 0.023385098204016685, -0.059375762939453125, ...]
#...
```
- embedding 모델에 따라 벡터 차원이 다르고 성능 및 효율이 다름

### 4. 검색기(Retriever) 생성

저장된 벡터 데이터베이스에서 사용자의 질문과 관련된 문서를 검색기 생성

```python
# 단계 5: 검색기(Retriever) 생성: 문서에 포함되어 있는 정보를 검색하고 생성합니다.

retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20}
)
```

- k: 리턴할 검색 문서 수
- fetch_k: 검색 문서 후보 수
- 검색 알고리즘 선택에 따라 성능 및 효율이 다름

![image](https://github.com/user-attachments/assets/16ff85c3-8abe-4880-ae6b-4ff34540c739)

### 5. LLM 입력 호출 

```python
    # 프롬프트를 생성
    prompt = load_prompt("src/assets/prompts/rag.yaml", encoding="utf-8")


    # 모델(LLM) 을 생성
    llm = ChatOpenAI(model_name="gpt-4o", temperature=1)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 사용자의 입력을 받아 답변 스트리밍 호출
    response = chain.stream("카카오모빌리티 광고상품에 대해 알려줘")

    # 카카오모빌리티 광고상품에 대해 알려줘 = [-0.01416763011366129, -0.022104470059275627, 0.0030067800544202328, ...]

    # document_chunked_1 = [0.02678847312927246, 0.03230374678969383, -0.02981565333902836, ...]
    # document_chunked_2 = [-0.020303381606936455, 0.015657879412174225, -0.059232909232378006, ...]
    # document_chunked_3 = [-0.005731410812586546, 0.023385098204016685, -0.059375762939453125, ...]
```

## 👀 RAG 모니터링

<img width="1266" alt="스크린샷 2024-11-25 오후 3 42 13" src="https://github.com/user-attachments/assets/1bf46391-3bf8-4836-848b-33ae72055fc9">

- RAG 프로세스 시퀀스가 어떻게 진행됬는지 확인 가능하며, 어디를 개선해야 할지 판단이 가능
- 사용된 토큰 개수와 과금 정보 확인 가능


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

OPENAI_API_KEY="<YOUR_API_KEY>" // https://platform.openai.com/
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="<YOUR_API_KEY>" // https://smith.langchain.com/
LANGCHAIN_PROJECT="<YOUR_PROJECT_NAME>"
```

### 5. 웹 애플리케이션 실행

Streamlit으로 웹 애플리케이션을 실행합니다.
```bash
streamlit run main.py
```
