from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from pydantic import BaseModel
from typing import Literal

from tavily import TavilyClient
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

class AgentState(TypedDict):
    query: str
    rewritten_query: str
    use_history: bool
    route: str
    docs: list
    answer: str
    check: bool
    not_grounded_explanation: str
    source: str
    retries: int
    web_retries: int
    chat_history: list
    no_docs: bool
    
    
class RouteDecision(BaseModel):
    route: Literal["rag", "chat", "end"]
    
class ContextDecision(BaseModel):
    use_history: bool
    
class GroundingCheck(BaseModel):
    grounded: bool
    explanation: str
    

    
llm = ChatOpenAI(
    model="x-ai/grok-4.1-fast",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0,
)


embeddings = OpenAIEmbeddings(
    model="qwen/qwen3-embedding-8b",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

vectorstore = Chroma(
    collection_name="pdf_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


##############



tavily = TavilyClient(api_key=TAVILY_API_KEY)

def web_search(query: str):
    response = tavily.search(
        query=query,
        max_results=3,
    )

    docs = []
    for r in response["results"]:
        docs.append(
            Document(
                page_content=r["content"],
                metadata={"source": r["url"]}
            )
        )
    print('\n\ndocs web search', docs)
    return docs

def web_search_node(state: AgentState):
    docs = web_search(state["rewritten_query"])
    return {
        "docs": docs,
        "source": "web",
        "web_retries": state.get("web_retries", 0) + 1, 
        "no_docs": False
    }

#############

def router_node(state):
    llm_structured = llm.with_structured_output(RouteDecision)

    result = llm_structured.invoke(f"""
    Decide the intent:

    Examples:
    Q: "What is LangChain?"
    → rag

    Q: "hi how are you"
    → chat

    Q: "bye"
    → end

    Rules:
    - If factual → rag
    - If conversational → chat
    - If unclear → rag
    - If user wants to stop -> end

    Query:
    {state["rewritten_query"]}
    """)
    print('\n\nroute', result.route)
    return {"route": result.route}

def retrieve_node(state: AgentState):
    docs = retriever.invoke(state["rewritten_query"])
    print('\n\ndocs', docs)
    return {
    "docs": docs,
    "source": "vectorstore",
    "retries": state.get("retries", 0) + 1
}
    
def rerank_node(state: AgentState):
    import requests

    docs = state["docs"][:10]

    if not docs:
        return {
            "docs": [],
            "no_docs": True
        }

    response = requests.post(
        "https://openrouter.ai/api/v1/rerank",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "cohere/rerank-4-fast",
            "query": state["rewritten_query"],
            "documents": [d.page_content for d in docs],
            "top_n": 4,
        },
    )

    if response.status_code != 200:
        return {"docs": docs[:4]}

    data = response.json()

    reranked = [docs[r["index"]] for r in data["results"]]
    print(f'\n\nreranked', data)
    return {"docs": reranked}

def generate_node(state: AgentState): 
    context = "\n\n".join([
    f"[{d.metadata.get('chunk_id', d.metadata.get('source', 'web'))}]: {d.page_content}"
    for d in state["docs"]
    ])

    response = llm.invoke(f"""
    Answer the question using ONLY the provided context.

    - Cite sources using [chunk_id]
    - If the answer is not in the context, say "I don't know"

    Question:
    {state['rewritten_query']}

    Context:
    {context}
    
    If context is empty → return "I don't know"
    
    """)

    return {"answer": response.content}


def self_check_node(state):
    if state["answer"] == "I don't know":
        return {"check": False, "not_grounded_explanation": "not enough context"}
    
    llm_structured = llm.with_structured_output(GroundingCheck)

    context = " ".join([d.page_content for d in state["docs"]])

    result = llm_structured.invoke(f"""
    Is the answer fully supported by the context?
    
    Return:
    - grounded: true/false.
    - explanation: short reason

    Answer:
    {state["answer"]}

    Context:
    {context}
    """)
    print('\n\ncheck', result.grounded)
    return {"check": result.grounded, "not_grounded_explanation": result.explanation}


def check_grounding(state):
    if state.get("check"):
        return "done"
    
    if state.get("retries", 0) >= 3 or state.get("web_retries", 0) >= 2:
        return "done"

    if state.get("no_docs"):
        return "web"

    source = state.get("source")
    if source == "vectorstore":
        return "web" 
    
    if source == "web":
        return "web"

    return "done"

def context_check_node(state):
    llm_structured = llm.with_structured_output(ContextDecision)

    result = llm_structured.invoke(f"""
    Does this query depend on previous conversation?

    Chat history:
    {state.get("chat_history", [])[-3:]}

    Query:
    {state["rewritten_query"]}
    """)
    print('\n\nuse history', result.use_history)
    return {"use_history": result.use_history}

def rewrite_node(state: AgentState):
    if not state["use_history"]:
        return {"rewritten_query": state["query"]}

    response = llm.invoke(f"""
    Rewrite into a standalone, specific search query.
    Do not lose important details.

    Chat history:
    {state['chat_history'][-3:]}

    Query:
    {state['query']}
    """)
    print('\n\nrewritten query', response.content)
    return {"rewritten_query": response.content}

def chat_node(state: AgentState):
    response = llm.invoke(f"""
    Continue the conversation naturally.

    Chat history:
    {state['chat_history']}

    User:
    {state['query']}
    """)
    print('\n\nanswer', response.content)
    return {"answer": response.content}

def route_decision(state: AgentState):
    return state["route"]
    
    

from langgraph.graph import StateGraph, START, END

builder = StateGraph(AgentState)


builder.add_node("context_check", context_check_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("router", router_node)

builder.add_node("retrieve", retrieve_node)
builder.add_node("web_search", web_search_node)
builder.add_node("rerank", rerank_node)
builder.add_node("generate", generate_node)
builder.add_node("self_check", self_check_node)

builder.add_node("chat", chat_node)


builder.add_edge(START, "context_check")
builder.add_edge("context_check", "rewrite")
builder.add_edge("rewrite", "router")

builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "rag": "retrieve",
        "chat": "chat",
        "end": END,
    },
)

# RAG pipeline
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "generate")

builder.add_conditional_edges(
    "rerank",
    lambda s: "no_docs" if s.get("no_docs") else "ok",
    {
        "no_docs": "web_search",
        "ok": "generate",
    },
)

builder.add_edge("web_search", "generate")
# builder.add_edge("web_search", "rerank")
builder.add_edge("generate", "self_check")

builder.add_conditional_edges(
    "self_check",
    check_grounding,
    {
        "retry": "retrieve",
        "web": "web_search",
        "done": END,
    },
)

# Chat ends directly
builder.add_edge("chat", END)

graph = builder.compile()

# print(graph.get_graph().print_ascii())


if __name__ == '__main__':
    chat_history = []
    
    while True:
        user_input = input("User: ")

        result = graph.invoke({
                "query": user_input,
                "rewritten_query": user_input,
                "chat_history": chat_history,
                "retries": 0,
                "web_retries": 0,
                "docs": [],
                "no_docs": False,
            })

        answer = result.get('answer')
    
        if answer:
            print("AI:", answer)
        else:
            break

        chat_history.append((user_input, answer))