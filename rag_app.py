import os
from typing import TypedDict, Annotated, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage
)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# ============================================================
# 1. EMBEDDING MODEL
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 2. LOAD PRE-BUILT VECTOR STORES
# ============================================================

academic_vectorstore = FAISS.load_local(
    "vectorstores/academic",
    embeddings,
    allow_dangerous_deserialization=True
)

fee_vectorstore = FAISS.load_local(
    "vectorstores/fee",
    embeddings,
    allow_dangerous_deserialization=True
)

academic_retriever = academic_vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

fee_retriever = fee_vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


# ============================================================
# 3. LLM
# ============================================================

llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0.2
)


# ============================================================
# 4. DYNAMIC ROUTING SCHEMA
# ============================================================

class RouteDecision(BaseModel):

    category: Literal[
        "academic",
        "fee",
        "general"
    ] = Field(
        description=(
            "Choose 'academic' for courses/exams/rules, "
            "'fee' for tuition/money/refunds, or "
            "'general' for casual/greetings/other."
        )
    )

    search_query: str = Field(
        description=(
            "Optimized standalone keyword query for document "
            "retrieval incorporating the course/programme if known."
        )
    )

    detected_programme: Optional[str] = Field(
        default=None,
        description=(
            "Extract any course or programme mentioned by the user "
            "(e.g., 'BCA', 'B.Tech CSE', 'MBA', 'BBA', 'B.Com'). "
            "If no course is mentioned in the conversation, return None."
        )
    )


structured_router = llm.with_structured_output(RouteDecision)


# ============================================================
# 5. GRAPH STATE
# ============================================================

class GraphState(TypedDict):

    programme: Optional[str]

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]

    route: RouteDecision

    retrieved_context: str


# ============================================================
# 6. CLASSIFIER & INTENT NODE
# ============================================================

def classify_and_rewrite_node(state: GraphState) -> dict:

    history = state["messages"]

    prompt = [
        SystemMessage(
            content=(
                "You analyze student inquiries. Classify the intent into 'academic', "
                "'fee', or 'general'. Extract any academic degree or programme the "
                "user mentions, and produce an optimized standalone search query "
                "incorporating context from the chat history."
            )
        ),
        *history
    ]

    decision = structured_router.invoke(prompt)

    # Retain previously learned programme if none was mentioned in this turn
    existing_programme = state.get("programme")
    resolved_programme = decision.detected_programme or existing_programme

    return {
        "route": decision,
        "programme": resolved_programme
    }


# ============================================================
# 7. ACADEMIC RETRIEVAL NODE
# ============================================================

def academic_rag_node(state: GraphState) -> dict:

    query = state["route"].search_query
    docs = academic_retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    return {
        "retrieved_context": context
    }


# ============================================================
# 8. FEE RETRIEVAL NODE
# ============================================================

def fee_rag_node(state: GraphState) -> dict:

    query = state["route"].search_query
    docs = fee_retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    return {
        "retrieved_context": context
    }


# ============================================================
# 9. GENERAL NODE
# ============================================================

def general_node(state: GraphState) -> dict:

    return {
        "retrieved_context": "NONE"
    }


# ============================================================
# 10. RESPONSE NODE
# ============================================================

def response_node(state: GraphState) -> dict:

    programme = state.get("programme")
    context = state.get("retrieved_context", "NONE")

    student_label = f"enrolled in {programme}" if programme else "a student"

    if context == "NONE":
        system_text = (
            f"You are a friendly, universal college assistant speaking with {student_label}. "
            f"Answer conversationally and helpfully."
        )
    else:
        system_text = (
            f"You are a helpful college assistant assisting {student_label}.\n\n"
            f"Use ONLY the verified context below to answer questions about college "
            f"rules, academics, fees, courses, exams, and policies.\n\n"
            f"Guidelines:\n"
            f"1. If specific rules or figures vary by course/programme and the student's "
            f"course is unknown, clarify what varies or politely ask for their specific course.\n"
            f"2. If the answer is not present in the context, clearly state that the "
            f"provided documents do not contain the answer.\n\n"
            f"Context:\n{context}"
        )

    messages = [
        SystemMessage(content=system_text)
    ] + state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


# ============================================================
# 11. ROUTING
# ============================================================

def route_next(state: GraphState) -> str:

    category = state["route"].category

    if category == "academic":
        return "academic_rag"

    if category == "fee":
        return "fee_rag"

    return "general"


# ============================================================
# 12. BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(GraphState)

workflow.add_node("classifier", classify_and_rewrite_node)
workflow.add_node("academic_rag", academic_rag_node)
workflow.add_node("fee_rag", fee_rag_node)
workflow.add_node("general", general_node)
workflow.add_node("response", response_node)

workflow.add_edge(START, "classifier")

workflow.add_conditional_edges(
    "classifier",
    route_next,
    {
        "academic_rag": "academic_rag",
        "fee_rag": "fee_rag",
        "general": "general"
    }
)

workflow.add_edge("academic_rag", "response")
workflow.add_edge("fee_rag", "response")
workflow.add_edge("general", "response")
workflow.add_edge("response", END)


# ============================================================
# 13. CHECKPOINTER
# ============================================================

checkpointer = MemorySaver()

app = workflow.compile(
    checkpointer=checkpointer
)


# ============================================================
# 14. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "session_user_live"
        }
    }

    print("College Assistant is ready. Ask anything! (Type 'exit' to quit)\n")

    while True:

        user_msg = input("You: ").strip()

        if user_msg.lower() in ["exit", "quit"]:
            break

        if not user_msg:
            continue

        result = app.invoke(
            {
                "messages": [
                    HumanMessage(content=user_msg)
                ]
            },
            config=config
        )

        print(f"\nAssistant: {result['messages'][-1].content}\n")