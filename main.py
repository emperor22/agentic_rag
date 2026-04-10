from langgraph.graph import StateGraph, START, END
from schema import AgentState
from nodes import (context_check_node, rewrite_node, router_node, retrieve_node, web_search_node, 
                   rerank_node, generate_node, self_check_node, chat_node, route_decision, check_grounding
)

def create_graph():
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
    
    builder.add_edge("chat", END)

    graph = builder.compile()
    
    return graph


if __name__ == '__main__':
    graph = create_graph()
    
    # print(graph.get_graph().print_ascii())
    
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