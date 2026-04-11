
from schema import AgentState, RouteDecision, GroundingCheck, ContextDecision
from logger import logger


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from tavily import TavilyClient
from langchain_core.documents import Document

from config import config

    
llm = ChatOpenAI(
    model=config.PRIMARY_MODEL,
    base_url=config.BASE_URL,
    api_key=config.OPENROUTER_API_KEY,
    temperature=0,
)


embeddings = OpenAIEmbeddings(
    model=config.EMBEDDING_MODEL,
    base_url=config.BASE_URL,
    api_key=config.OPENROUTER_API_KEY,
)

vectorstore = Chroma(
    collection_name=config.COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=config.CHROMA_PERSIST_DIR,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})

tavily = TavilyClient(api_key=config.TAVILY_API_KEY)



def web_search(query: str):
    response = tavily.search(
        query=query,
        max_results=config.TAVILY_RESULTS,
    )

    docs = []
    for r in response["results"]:
        docs.append(
            Document(
                page_content=r["content"],
                metadata={"source": r["url"]}
            )
        )

    return docs

def web_search_node(state: AgentState):
    docs = web_search(state["rewritten_query"])
    return {
        "docs": docs,
        "source": "web",
        "web_retries": state.get("web_retries", 0) + 1, 
        "no_docs": False
    }

def is_end_signal(text: str):
    text = text.lower().strip()

    end_phrases = [
        "thanks", "thank you", "thx", "ok thanks", "got it",
        "understood", "ok understood", "makes sense",
        "bye", "goodbye", "see you", "that helps", "all good"
    ]

    return any(p in text for p in end_phrases)


def router_node(state):
    query = state["rewritten_query"]

    if is_end_signal(query):
        logger.success("Decision: Route to -> END (rule-based)")
        return {"route": "end"}

    llm_structured = llm.with_structured_output(RouteDecision)

    result = llm_structured.invoke(f"""
You are an intent classifier.

Return ONLY one of: end, rag, chat.

Strict rules:
- "end" → user expresses completion, satisfaction, or closing intent
  (e.g., thanks, thank you, ok, understood, got it, makes sense, bye)
- "rag" → factual question or knowledge retrieval
- "chat" → casual conversation or small talk

Important:
- If the message contains gratitude or acknowledgement WITHOUT a new question → end
- If unsure between end and anything else → end
- Do NOT choose rag unless a real question is asked

Examples:
"bye" → end
"ok thanks" → end
"understood, thank you" → end
"that helps" → end

"What is LangChain?" → rag
"Explain vector databases" → rag

"hi how are you" → chat
"nice weather today" → chat

Query:
{query}
""")

    logger.success(f"Decision: Route to -> {result.route.upper()}")
    return {"route": result.route}

def retrieve_node(state: AgentState):
    docs = retriever.invoke(state["rewritten_query"])
    return {
    "docs": docs,
    "source": "vectorstore",
    "retries": state.get("retries", 0) + 1
}
    
def rerank_node(state: AgentState):
    import requests

    docs = state["docs"][:config.RETRIEVER_K]

    if not docs:
        return {
            "docs": [],
            "no_docs": True
        }

    response = requests.post(
        config.BASE_URL_RERANK,
        headers={
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.RERANK_MODEL,
            "query": state["rewritten_query"],
            "documents": [d.page_content for d in docs],
            "top_n": config.RERANK_TOP_N,
        },
    )

    if response.status_code != 200:
        return {"docs": docs[:config.RERANK_TOP_N]}

    data = response.json()

    reranked = [docs[r["index"]] for r in data["results"]]
    
    logger.success(f"Rerank complete. Kept {len(data['results'])} high-relevance chunks.")
    
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


def self_check_node(state: AgentState):
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
    
    if result.grounded:
        logger.success("Grounding Check: PASSED")
    else:
        logger.warning(f"Grounding Check: FAILED. Reason: {result.explanation}")
    
    return {"check": result.grounded, "not_grounded_explanation": result.explanation}


def check_grounding(state: AgentState):
    if state.get("check"):
        return "done"
    
    if (state.get("retries", 0) >= config.MAX_RAG_RETRIES or 
        state.get("web_retries", 0) >= config.MAX_WEB_RETRIES):
        return "done"

    if state.get("no_docs"):
        return "web"

    source = state.get("source")
    if source == "vectorstore":
        return "web" 
    
    if source == "web":
        return "web"

    return "done"

def context_check_node(state: AgentState):
    llm_structured = llm.with_structured_output(ContextDecision)

    result = llm_structured.invoke(f"""
    Does this query depend on previous conversation?

    Chat history:
    {state.get("chat_history", [])[-config.CHAT_HISTORY_WINDOW:]}

    Query:
    {state["query"]}
    """)
    return {"use_history": result.use_history}

def rewrite_node(state: AgentState):
    if not state["use_history"]:
        return {"rewritten_query": state["query"]}

    response = llm.invoke(f"""
    Rewrite into a standalone, specific search query.
    Do not lose important details.

    Chat history:
    {state['chat_history'][-config.CHAT_HISTORY_WINDOW:]}

    Query:
    {state['query']}
    """)
    
    logger.info(f"Query Refinement: [{state['query']}] -> {response.content}")
    
    return {"rewritten_query": response.content}

def chat_node(state: AgentState):
    response = llm.invoke(f"""
    Continue the conversation naturally.

    Chat history:
    {state['chat_history']}

    User:
    {state['query']}
    """)

    return {"answer": response.content}

def route_decision(state: AgentState):
    return state["route"]
    