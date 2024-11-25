from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


def get_retreiver():
    # # 단계 1: 문서 로드(Load Documents)
    # loader = PyMuPDFLoader("data/SPRI_AI_Brief_2023년12월호_F.pdf")
    # docs = loader.load()

    # # 단계 2: 문서 분할(Split Documents)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    # split_documents = text_splitter.split_documents(docs)

    # 단계 1: 여러 파일 로드 및 분할 처리
    # 파일 경로 리스트
    file_paths = [
        "src/assets/txt/AD_INFO.txt",
        "src/assets/txt/AD_DA.txt",
        "src/assets/txt/AD_RSE.txt",
        "src/assets/txt/AD_OOH.txt",
        "src/assets/txt/AD_CS.txt",
        "src/assets/txt/AD_REWORD.txt",
        "src/assets/txt/AD_ALL.txt",
        "src/assets/txt/BIZ_CENTER.txt",
    ]
    all_split_documents = []

    # 단계 2: 각 파일에 대해 문서를 로드하고 분할
    for file_path in file_paths:
        # 단계 1: 문서 로드
        # loader = PyMuPDFLoader(file_path)
        # docs = loader.load()

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

            document = Document(page_content=text)

        # 단계 2: 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=100
        )
        split_documents = text_splitter.split_documents([document])

        print(
            f"Total split documents: {len(all_split_documents)}, {len(split_documents)}"
        )
        # 분할된 문서 청크를 리스트에 추가
        all_split_documents.extend(split_documents)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(
        documents=all_split_documents, embedding=embeddings
    )

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20}
    )

    # 사용자가 입력한 쿼리의 의미를 다각도로 포착하여 검색 효율성을 높이고, LLM을 활용하여 사용자에게 보다 관련성 높고 정확한 정보를 제공하는 것을 목표
    llm = ChatOpenAI(model_name="gpt-4o", temperature=1)
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever, llm=llm)

    # 컨텍스트 압축 기법은 검색된 문서 중에서 쿼리와 관련된 정보만을 추출하여 반환하는 것을 목표로 합니다.
    # 기본 검색기를 통해 얻은 문서들을 문서 압축기를 사용하여 내용을 압축하고, 쿼리와 가장 관련된 내용만을 추려냅니다.
    # compressor = LLMChainExtractor.from_llm(llm)
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor, base_retriever=retriever_from_llm
    # )

    return retriever_from_llm
