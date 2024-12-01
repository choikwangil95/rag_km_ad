from src.functions.memory import get_session_history
from src.functions.retriever import get_retreiver
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import load_prompt
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory

# 문서 검색 Agent
retriever = get_retreiver()
retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_search",  # 도구의 이름을 입력합니다.
    description="use this tool to search information from the PDF document",  # 도구에 대한 설명을 자세히 기입해야 합니다!!
)


# 웹 검색 Agent
search_tool = TavilySearchResults(k=6)


# Supervisor Agent
def get_agent_superviser():
    # 마이크로 Agent 목록
    tools = [retriever_tool, search_tool]
    # 프롬프트
    prompt = load_prompt("src/assets/prompts/agent.yaml", encoding="utf-8")
    # LLM 모델
    llm = ChatOpenAI(model_name="gpt-4o", temperature=1)

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    # 채팅 메시지 기록이 추가된 에이전트를 생성합니다.
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # 대화 session_id
        get_session_history,
        # 프롬프트의 질문이 입력되는 key: "input"
        input_messages_key="input",
        # 프롬프트의 메시지가 입력되는 key: "chat_history"
        history_messages_key="chat_history",
    )

    return agent_with_chat_history
